import redis
import os
from azure.identity import DefaultAzureCredential
import time
import hashlib
import logging
import json
from func_timeout import func_timeout
from normalization import get_normalized_sentence
class CacheOptions:
    def __init__(self):
        self.cache_flag = os.getenv("Enable_Cache").lower()
        self.cache_log_flag = os.getenv("Cache_Debug").lower()
        self.timeout = int(os.getenv("Timeout")) * 0.001
        self.expiration_time = os.getenv("Cache_expiration_time")
class RedisCache:
    def __init__(self, cache_options):
        self.cache_options = cache_options
        self.redis_connect = self.get_redis_connection()
    def get_redis_connection(self):
        retry_times = 3
        retry_wait_time = 10 #in second
        for retry_count in range(retry_times):
            try:
                creds = DefaultAzureCredential()
                scope = os.getenv("Credential_Scope")
                token = creds.get_token(scope)
                user_name = os.getenv("Redis_user_name")
                redis_host = os.getenv("Redis_Cache")
                redis_port = 6380
                r = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    ssl=True,
                    username=user_name,
                    password=token.token,
                    decode_responses=True
                )
                logging.error('########## Redis Cache Host Connection Intialized ##########')
                return r
            except Exception as e:
                logging.error("Failed to connect to Redis cache:", str(e))
                if retry_count < retry_times - 1:
                    logging.error(f"Retrying after {retry_wait_time} seconds...")
                    time.sleep(retry_wait_time)
                else:
                    break
        return None

    def GenerateCacheKey(self, src_lang, tgt_lang, source_fast_words, orig_tgt_fast_words):
        combined_data = src_lang + tgt_lang + " ".join(source_fast_words) + " ".join(orig_tgt_fast_words)
        print('combined_data:',combined_data)
        combined_data_bytes = combined_data.encode('utf-8')
        print('combined_data_bytes:',combined_data_bytes)
        hash_data = hashlib.sha256(combined_data_bytes).hexdigest()
        return hash_data

    def normalized_sentence(self, src_text, tgt_text):
        source_fast_words, orig_tgt_fast_words,space_norm_source, space_norm_orig_tgt = get_normalized_sentence(src_text, tgt_text)
        return source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt

    def get_data_from_cache(self, r, key, trace_id):
        try:
            def get_data():
                start_time = time.time()
                value = r.get(key)
                latency = round((time.time() - start_time) * 1000, 2)
                return value, latency
            try:
                print("getenvTimeout:", int(os.getenv("Timeout")))
                value, latency = func_timeout(int(os.getenv("Timeout")) * 0.001, get_data) 
                print("value:",value)
                print('key:',key)
                if value is not None:
                    return value, latency
                else:
                    return None, None
            except Exception as e:
                logging.error("get data in cache call timed out: %s (trace id: %s)", str(e),str(trace_id))
                return None, None
        except Exception as e:
            logging.error("Failed to get data in cache: %s, trace id is: %s", str(e), str(trace_id))
            return None, None

    def set_data_in_cache(self, r, key, value, trace_id, expiration_time=None):
        try:
            def set_data():
                start_time = time.time()
                if expiration_time is not None:
                    r.setex(key, expiration_time, value)
                else:
                    r.set(key, value)
                latency = round((time.time() - start_time) * 1000, 2)
                return latency
            try:
                latency = func_timeout(int(os.getenv("Timeout")) * 0.001, set_data) 
                return latency
            except Exception as e:
                logging.error("set data in cache call timed out: %s (trace id: %s)", str(e),str(trace_id))
                return None
        except Exception as e:
            logging.error("Failed to set data in cache: %s, trace id is: %s", str(e), str(trace_id))
            return None


    def try_get_entry_from_cache(self, aml_request, redis_connect,trace_id,cache_flag):
        source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt = self.normalized_sentence(aml_request.src_text, aml_request.tgt_text)
        if cache_flag == "true":
            src_lang_str = str(aml_request.src_lang)
            tgt_lang_str = str(aml_request.tgt_lang)
            if redis_connect:
                cache_key = self.GenerateCacheKey(src_lang_str, tgt_lang_str, source_fast_words, orig_tgt_fast_words)
                print('cache_key:',cache_key)
                cached_response, get_latency = self.get_data_from_cache(redis_connect, cache_key,trace_id)
                if cached_response:
                    cached_response_valid_json = cached_response.replace("'", "\"")
                    cached_response_obj = json.loads(cached_response_valid_json)
                    updated_cached_response = {"src_sentence": aml_request.src_text, "tgt": cached_response_obj}
                    updated_cached_response_str = json.dumps(updated_cached_response)
                    return updated_cached_response_str, get_latency, cache_key,source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt
                return None, None, cache_key,source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt
            logging.error("Cache Get: No Cache Redis Connection was established.")
        return None, None, None, source_fast_words, orig_tgt_fast_words, space_norm_source, space_norm_orig_tgt

    def try_set_entry_from_cache(self,updated_cached_response_str, redis_connect, cache_key, tgt_dict, trace_id,
                                 cache_flag):
        if cache_flag == "true":
            if redis_connect:
                if updated_cached_response_str is None:
                    tgt_part_str = str(tgt_dict)
                    set_latency = self.set_data_in_cache(redis_connect, cache_key, tgt_part_str, trace_id, os.getenv("Cache_expiration_time"))
                    return set_latency
            else:
                logging.error("Cache Set: No Cache Redis Connection was established.")
        else:
            logging.error("Cache not enabled or not an online deployment.")
        return None
