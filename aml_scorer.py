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
from redis_cache import *

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

def get_hostname_cpu():
    cpu_type_command = "cat /proc/cpuinfo"
    cpu_all_info = subprocess.check_output(cpu_type_command, shell=True).decode().strip()
    cpu_useful_info = '\t'.join(cpu_all_info.replace('\t','').split('\n')[:9])
    return cpu_useful_info

def init():
    global redis_connect
    # Make sure the model version is in sync with gender_debias_pipeline/AML-config/blue-deployment-azure.yml
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "modelfiles")
    config_path = os.path.join(model_path, "default_config.json")
    log_path = os.path.join(model_path, "logging_config.json")

    global logging_config
    logging_config = load_json_file(log_path)

    global app_logger

    if os.getenv("LOCAL_DEPLOYMENT").lower() == "true":
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
    if os.getenv("LOCAL_DEPLOYMENT").lower() == "false":
        redis_connect = get_redis_connection()

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
        request_logger.info(f"TIME: {now}")
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
        #Cache Entry
        if os.getenv("LOCAL_DEPLOYMENT").lower() == "false":
            src_lang_str = str(aml_request.src_lang)
            tgt_lang_str = str(aml_request.tgt_lang)
            cache_key = GenerateCacheKey(src_lang_str, tgt_lang_str, aml_request.src_text, aml_request.tgt_text)
            cache_log_flag = os.getenv("Cache_Debug")
            if redis_connect is not None:
                cached_response, get_latency = get_data_from_cache(redis_connect, cache_key)
                if cached_response:
                    if cache_log_flag.lower() == "true":
                        logger.info("Retrieved response from cache, latency is: %.2f milliseconds" % get_latency)
                        cached_response_obj = json.loads(cached_response)
                        cached_response_obj["src_sentence"] = aml_request.src_text
                        cached_response_obj["tgt"] = tgt_dict
                        updated_cached_response = get_json(cached_response_obj)
                        return AMLResponse(updated_cached_response, 200, aml_request.response_headers)
        start_time = datetime.now()
        # with tracer.span(name='Debias_Models.get_reinflection_single_sentence'):
        reinflection_result = model.get_reinflection_single_sentence(aml_request.src_text, aml_request.tgt_text, verbose=True, max_words=aml_request.options.max_words, max_hypotheses=aml_request.options.max_hypotheses, request_logger=request_logger)
        
        request_logger.info(f"reinflection complete. Time taken = {datetime.now() - start_time}")
        request_logger.info(f'########## SCORE END ##########')
        now = datetime.now()
        request_logger.info(f"TIME: {now}")
        has_reinflection = False

        test_request = False    
        #check is the header contains the string API-TEST and if so don't track it.
        if "API-TEST" in aml_request.response_headers[client_traceid_response_header_name]:
            test_request = True
        
        if not test_request:
            tmap_request = tag_map_module.TagMap()
            tmap_request.insert("srcLanguage", aml_request.src_lang.value)
            tmap_request.insert("tgtLanguage", aml_request.tgt_lang.value)
            requests_view_measurement_map.measure_int_put(number_of_requests_measure, 1)
            requests_view_measurement_map.record(tmap_request)

        if not reinflection_result.has_reinflection():
            # reinflection was aborted for some reason
            logger.debug (f"No reinflection. Reason = {reinflection_result.aborted_reason} ({trace_id})")
            tgt_dict = {str(ApiGender.Neutral) : aml_request.tgt_text}
        else:
            if not test_request:
                tmap_reinflections = tag_map_module.TagMap()
                tmap_reinflections.insert("srcLanguage", aml_request.src_lang.value)
                tmap_reinflections.insert("tgtLanguage", aml_request.tgt_lang.value)
                reinflections_view_measurement_map.measure_int_put(number_of_reinflections_measure, 1)
                reinflections_view_measurement_map.record(tmap_reinflections)

            has_reinflection = True
            best_hyp_gender = reinflection_result.get_best_hyp_gender()
            best_hyp = reinflection_result.get_best_hyp()
            if best_hyp_gender == Gender.Ambiguous:
                request_logger.debug(f"gender of the hypothesis was not clear, assuming orig_tgt is Masculine and reinflection is Feminine ({trace_id})")

            masc_tgt = best_hyp             if best_hyp_gender == Gender.Male else aml_request.tgt_text
            fem_tgt  = aml_request.tgt_text if best_hyp_gender == Gender.Male else best_hyp
            
            tgt_dict = {
                str(ApiGender.Feminine): fem_tgt,
                str(ApiGender.Masculine): masc_tgt,
            }

        result = GenderDebiasResponse(aml_request.src_text, tgt_dict)
        if(aml_request.options.debug):
            result =  GenderDebiasDebugResponse(aml_request.src_text, tgt_dict, reinflection_result.debug_options())
        api_response = get_json(result)

        if log_input:
            request_logger.info(f"has_reinflection={has_reinflection}, Api_response={api_response}")

        # Set in Cache
        if os.getenv("LOCAL_DEPLOYMENT").lower() == "false":
            if cached_response is None:
                tgt_part_str = str(tgt_dict)
                set_latency = set_data_in_cache(redis_connect, cache_key, tgt_part_str, os.getenv("Cache_expiration_time"))
                if cache_log_flag == "true": 
                    logger.info("Set request response to cache with expiration time, latency is: %.2f milliseconds" % set_latency)
        return AMLResponse(api_response, 200, aml_request.response_headers)
    except Exception as e:
        error_code = 50000
        request_logger.info(f"Unexpected exception {traceback.format_exc()}. Errorcode:{error_code}")
        api_response = get_json(GenderDebiasErrorResponse(error_code, "Internal Server Error"))
        return AMLResponse(api_response, 500, aml_request.response_headers)

def try_match_sentfix(sentfix_manager, src, orig_tgt):
        sentfix_result = sentfix_manager.try_match_sentfix(src)
        if sentfix_result is None:
            return None
        
        if sentfix_result.is_neutral():
            tgt = {str(ApiGender.Neutral): sentfix_result.fem_trans} # fem and masc translations are equivalent
        elif sentfix_result.is_orig_passthrough():
            tgt = {str(ApiGender.Neutral): orig_tgt}                 # indicates we should not reinflect, just use MT output
        else:
            tgt =  {
                        str(ApiGender.Feminine): sentfix_result.fem_trans,
                        str(ApiGender.Masculine): sentfix_result.masc_trans
                   }

        return GenderDebiasResponse(src, tgt)

def local_web_service_testing():
    init()
    data = """{
                "source": {
                    "language": "en",
                    "text": "The cook is making dinner."
                },
                "target": {
                    "language": "es",
                    "text": "El cocinero está preparando la cena."
                }
            }"""
    response = run(data)
    print(f'response={response}')

if __name__ == "__main__":
    local_web_service_testing()
