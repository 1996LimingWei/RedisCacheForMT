from redis_cache import CacheOptions, RedisCache
redis_connect = None
redis_cache = None
def init():
    global redis_connect
    global cache_flag
    global local_deployment_flag
    global redis_cache
    cache_flag = os.getenv("enable_cache").lower()
    if cache_flag == "true":
        cache_options = CacheOptions()
        redis_cache = RedisCache(cache_options)
        redis_connect = redis_cache.get_redis_connection()
def run(data):
    global redis_cache
    global redis_connect
        #Cache Entry 
        cached_response_obj, cache_key, new_connect, new_connect_bool = redis_cache.try_get_entry_from_cache(aml_request, redis_connect, trace_id,logger, cache_log_flag,
                                                                                        source_fast_words, orig_tgt_fast_words)
        if new_connect_bool:
            if redis_connect is not None:
                try:
                    redis_connect.close()
                except Exception as e:
                    logger.info(f"Failed to close old Redis connection: {e}")
            redis_connect = new_connect
            logger.info(f'redis connect re-establish')
        if cached_response_obj:
            time00 = round((time.time() - time_start) * 1000, 2)
            response_instance = GenderDebiasResponse(aml_request.src_text, cached_response_obj)
            api_response = get_json(response_instance)
            logger.info(f'Get data from cache total time is {time00} milliseconds, text: {aml_request.src_text}')
            return AMLResponse(api_response, 200, aml_request.response_headers)

Here is the redis_cache.py file that has the redis_cache class:
class CacheOptions:
    def __init__(self):
        self.cache_flag = os.getenv("enable_cache").lower()
        
class RedisCache:
    def __init__(self, cache_options):
        self.cache_options = cache_options
        self.connection_lock = threading.Lock()        
    def get_redis_connection(self):
        print("retry times:", os.getenv("retry_times"))
        print("retry_wait_time:", os.getenv("retry_wait_time"))
        retry_times = 2
        retry_wait_time = 5
        for retry_count in range(retry_times):
            try:
                creds = DefaultAzureCredential()
                scope = os.getenv("credential_scope").lower()
                token = creds.get_token(scope)
                user_name = os.getenv("redis_user_name").lower()
                redis_host = os.getenv("redis_cache").lower()
                redis_port = 6380
                r = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    ssl=True,
                    username=user_name,
                    password=token.token,
                    decode_responses=True
                )
                logging.error(f'########## Redis Cache Host Connection Intialized ##########')
                return r
            except Exception as e:
                logging.error(f"Failed to connect to Redis cache: {str(e)}")
                if retry_count < retry_times - 1:
                    logging.error(f"Retrying after {retry_wait_time} seconds...")
                    time.sleep(retry_wait_time)
                else:
                    break
        return None
    def get_data_from_cache(self, r, key, trace_id, logger, aml_request_src_text):
        try:
            start_time = time.time()
            value = r.get(key)
            latency = round((time.time() - start_time) * 1000, 2)
            return value, latency, r, False
        except Exception as e:
            if "WRONGPASS" in str(e):
                logger.info(f"Authentication error occurred while accessing Redis cache: {str(e)}, text: {aml_request_src_text}, reconnecting......")
                with self.connection_lock:
                    r_new = self.get_redis_connection() 
                    try:
                        start_time = time.time()
                        value = r_new.get(key)
                        latency = round((time.time() - start_time) * 1000, 2)
                        return value, latency, r_new, True
                    except Exception as e:             
                        logger.info(f"Failed to get data from cache after reinitializing connection: {str(e)}, text: {aml_request_src_text}")
                        return None, None, None, False
            else:
                logger.info(f"get data in cache failed, error:{str(e)}, text: {aml_request_src_text}")
                return None, None, None, False
    def try_get_entry_from_cache(self, aml_request, redis_connect, trace_id, logger, cache_log_flag, source_fast_words, orig_tgt_fast_words):
        if aml_request.src_text == "Break the connection now for testing purpose sd4234hdfew4fdsw":
            logger.info("Break the connection now for testing purposes")
            redis_connect = None
        src_lang_str = str(aml_request.src_lang)
        tgt_lang_str = str(aml_request.tgt_lang)
        cache_key = self.GenerateCacheKey(src_lang_str, tgt_lang_str, source_fast_words, orig_tgt_fast_words)
        cached_response, get_latency, r, new_connect = self.get_data_from_cache(redis_connect, cache_key, trace_id, logger, aml_request.src_text)
        if cached_response:
            if cache_log_flag == "true":
                logger.info(f"get data in cache completed, latency is: {get_latency:.2f} milliseconds, text: {aml_request.src_text}")
                cached_response_obj = json.loads(cached_response)
            return cached_response_obj, cache_key, r, new_connect
        logger.info(f"get data in cache did not find any entry, text: {aml_request.src_text}")
