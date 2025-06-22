import os
from datetime import datetime
import re
import subprocess
import traceback
import uuid
from api_interfaces import *
import nltk
from model_initializer import load_all_models_from_json
from gender import Gender
from languages import Language, abbrev_to_lang
from azureml.contrib.services.aml_response import AMLResponse
from logger import AppLogger, get_disabled_logger
from gender_bias_utils import *
from datetime import datetime
from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module
from opencensus.tags import tag_map as tag_map_module
import json
from redis_cache import RedisCache

word_re = re.compile('\\w+', re.UNICODE)
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
component_name = "GenderDebias"

def full_file_path(filename):
    return os.path.join(script_dir, filename)


stats = stats_module.stats
view_manager = stats.view_manager
stats_recorder = stats.stats_recorder
##https://opencensus.io/stats/measure/
number_of_reinflections_measure = measure_module.MeasureInt("reinflections",
                                           "number of reinflections",
                                           "reinflections")
number_of_requests_measure = measure_module.MeasureInt("requests",
                                           "number of requests",
                                           "requests")

reinflections_view = view_module.View("reinflections view",
                               "number of reinflections",
                               ["srcLanguage","tgtLanguage"],
                               number_of_reinflections_measure,
                               aggregation_module.CountAggregation())

requests_view = view_module.View("requests view",
                               "number of requests",
                               ["srcLanguage","tgtLanguage"],
                               number_of_requests_measure,
                               aggregation_module.CountAggregation())
reinflections_view_measurement_map = stats_recorder.new_measurement_map()
requests_view_measurement_map = stats_recorder.new_measurement_map()
redis_connect = None
cache_flag = "false"
local_deployment_flag = "false"
def get_hostname_cpu():
    cpu_type_command = "cat /proc/cpuinfo"
    cpu_all_info = subprocess.check_output(cpu_type_command, shell=True).decode().strip()
    cpu_useful_info = '\t'.join(cpu_all_info.replace('\t','').split('\n')[:9])
    return cpu_useful_info

def init():
    global redis_connect
    global cache_flag
    global local_deployment_flag
    global redis_cache
    # Make sure the model version is in sync with gender_debias_pipeline/AML-config/blue-deployment-azure.yml
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "modelfiles")
    config_path = os.path.join(model_path, "default_config.json")
    log_path = os.path.join(model_path, "logging_config.json")

    global logging_config
    logging_config = load_json_file(log_path)

    global app_logger
    local_deployment_flag = os.getenv("LOCAL_DEPLOYMENT").lower()
    if local_deployment_flag == "true":
        app_logger = get_disabled_logger()
    else:
        app_insights_instrumentation_key = os.getenv("AML_APP_INSIGHTS_KEY")
        logging_config['app_insights_key'] = app_insights_instrumentation_key
        app_logger = AppLogger(config=logging_config)

    global logger
    logger = app_logger.get_logger(component_name=component_name)
    # tracing has big impact on CPU utilization. Hence, turn off tracing for now and investiagte custom metrics for the same.
    global tracer
    # tracer = app_logger.get_tracer(component_name=component_name)
    tracer = None
    cache_flag = os.getenv("Enable_Cache").lower()
    if cache_flag == "true":
        cache_options = CacheOptions()
        redis_cache = RedisCache(cache_options)
        redis_connect = redis_cache.get_redis_connection()

    logger.info('########## INIT Starting ########## v:4-07-2022 5:30PM')
    now = datetime.now()
    logger.info(f"TIME: {now}")

    global instance_id
    instance_id = uuid4()
    global machine_info
    machine_info = get_hostname_cpu()

    logger.info(f"showing init logs for instance_id {instance_id}, machine info: {machine_info}")

    # Set the nltk download path 
    nltk_data_path = os.path.join(model_path, "ivl", "nltk_data")
    logger.info(f"setting nltk data path:{nltk_data_path} with subfolders:{os.listdir(nltk_data_path)}")
    nltk.data.path.append(nltk_data_path)
    
    global Debias_Models
    # with tracer.span("load_all_models"):
    Debias_Models = load_all_models_from_json(config_path, app_logger=app_logger, parent_tracer=tracer)

    for lang in Debias_Models.keys():
        assert(Debias_Models[lang].is_fully_initialized())

    #register the metrics expoter
    metrics_exporter = app_logger.get_metrics_exporter()
    view_manager.register_exporter(metrics_exporter)
    view_manager.register_view(reinflections_view)
    view_manager.register_view(requests_view)

    logger.info('########## INIT END ########## v:2-15-2022 2:30PM')
    now = datetime.now()
    logger.info(f"TIME: {now}")	

standard_sample_input = {'srcLanguage': 'en', 'src': 'The doctor is going home as he is tired.', 'tgt': 'The doctor is going home as he is tired.' }
standard_sample_output = {'conllu': 'conllu format'}

class request_request_logger:
    def __init__(self, logger, trace_id):
        self.logger = logger
        self.trace_id = trace_id

    def add_trace_id(self, message):
        return f"{message} (Trace ID: {self.trace_id})"

    def info(self, message):
        logger.info(self.add_trace_id(message))

    def debug(self, message):
        logger.debug(self.add_trace_id(message))

def run(data):
    aml_request = None
    try:
        trace_id = str(uuid4()) 
        request_logger = request_request_logger(logger, trace_id)

        request_logger.info(f'########## SCORE START ##########')
        now = datetime.now()
        request_logger.info(f"Instance ID = {instance_id}, Machine info: {machine_info}")

        # validate and serialize request to aml_request object
        # with tracer.span("validate_request"):
        aml_request, errorResponse = validate_request(data, logger, trace_id)
        log_input = aml_request.options.log_input is True
        
        # return validation errors
        if(errorResponse is not None):
            return errorResponse
        
        model = Debias_Models[aml_request.tgt_lang]
        #return from hotfix if it has entries
        # with tracer.span(name='try_match_sentfix'):
        result = try_match_sentfix(model.sentfix_manager, aml_request.src_text, aml_request.tgt_text)
        if result is not None:
            api_response = get_json(result)
            if log_input:
                request_logger.info(f'Matched sentfix for {aml_request.src_text} -- api_response={api_response}')
            else:
                request_logger.info((f'Matched sentfix but log input disabled'))
            return AMLResponse(api_response, 200, aml_request.response_headers)
        cache_log_flag = os.getenv("Cache_Debug").lower()
        #Cache Entry 
        updated_cached_response_str, get_latency, cache_key, source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt = redis_cache.try_get_entry_from_cache(aml_request, redis_connect, trace_id, cache_flag)
        if updated_cached_response_str and cache_log_flag.lower() == "true":
            logger.info(f"get data in cache completed, latency is: {get_latency:.2f} milliseconds, text: {aml_request.src_text}")
            return AMLResponse(updated_cached_response_str, 200, aml_request.response_headers)

At this line:         updated_cached_response_str, get_latency, cache_key, source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt = redis_cache.try_get_entry_from_cache(aml_request, redis_connect, trace_id, cache_flag)
