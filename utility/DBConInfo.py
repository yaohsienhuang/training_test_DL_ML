import cx_Oracle
import pandas as pd
import time

class dbConInfo():
    def __init__(self,host, port, sid, acc, pw):
        self.__host = host
        self.__port = port
        self.__sid  = sid
        self.__acc  = acc
        self.__pw   = pw
        self.__dsn  = host + ':' + port + '/' + sid

    def select(self, sql):
        start_time = time.time()
        result = None
        log_head = '[%s][select]' % (self.__class__.__name__ )
        print(log_head+'Start')
        try:            
            db = cx_Oracle.connect(self.__acc, self.__pw, self.__dsn)
            print(log_head+'SQL=',sql)
            result = pd.read_sql(sql, db)
            print(log_head+'result_count =', result.shape[0])
        except Exception as e:
            print(log_head+'[Error]', e)
            raise Exception(e)
        finally:        
            db.close  
        end_time = time.time()
        print(log_head+'End, using time = %6.1f s' % (end_time-start_time))
        return result
    
    def execute(self, sql_list, show_sql = True):
        start_time = time.time()
        log_head = '[%s][execute]' % (self.__class__.__name__ )
        print(log_head+'Start')
        try:            
            db = cx_Oracle.connect(self.__acc, self.__pw, self.__dsn)            
            cursor = db.cursor() #建立遊標
            
            for i in range(len(sql_list)):
                sql = sql_list[i]
                if(show_sql):
                    print(log_head+'SQL=',sql)
                cursor.execute(sql)
            db.commit()
        except Exception as e:
            print(log_head+'[Error]', e)
            raise Exception(e)
        finally:        
            cursor.close()
            db.close()
        end_time = time.time()
        print(log_head+'End, using time = %6.1f s' % (end_time-start_time))
        
    def executemany(self, sql_fmt, data):
        start_time = time.time()
        #result = None
        log_head = '[%s][executemany]' % (self.__class__.__name__ )
        print(log_head+'Start')
        try:            
            db = cx_Oracle.connect(self.__acc, self.__pw, self.__dsn)            
            cursor = db.cursor() #建立遊標                       
            cursor.executemany(sql_fmt, data, batcherrors=True)
            for error in cursor.getbatcherrors():
                print("Error", error.message, "at row offset", error.offset)

            db.commit()
        except Exception as e:
            print(log_head+'[Error]', e)
            raise Exception(e)
        finally:        
            cursor.close()
            db.close()
        end_time = time.time()
        print(log_head+'End, using time = %6.1f s' % (end_time-start_time))

        
# 取第一片(抽測)圖片集
def get_first_wafer_image(start_date, end_date, model_name):
    sql = '''
    SELECT a.ENTITY, a.RECIPE, a.LOTID, a.WAFERID, a.FILENAME, a.OP_RESULT , a.AI_RESULT
    FROM AIDBA.WB_AI_IMAGE_INFO a, AIDBA.wb_ai_wafer_info b, EITEAP.AI_WAFER_INFORMATION@EAPWB c , EITEAP.AI_Wafer_detail@EAPWB d 
    WHERE (c.LOT_END_DATE between to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss') and to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss')) 
    AND c.WAFERID = d.WAFERID
    AND c.LOTID = d.LOTID
    AND c.ENTITY = d.ENTITY
    AND c.WAFERID = a.WAFERID
    AND c.LOTID = a.LOTID
    AND c.ENTITY = a.ENTITY
    AND c.WAFERID = b.WAFERID
    AND c.LOTID = b.LOTID
    AND c.ENTITY = b.ENTITY
    AND a.FILENAME = d.TIFF_FILENAME
    AND c.CUSTOMER != 'CM'
    AND (b.rework = 'N' or b.rework is null)
    AND b.blacklist = 'Y'
    AND c.slot = 1
    AND a.op_result is not null
    AND a.recipe not like '%%4588%%'
    AND a.recipe not like '%%FQC%%'
    AND a.recipe not like '%%IQC%%'
    AND a.recipe not like '%%BV%%'
    AND a.recipe not like '%%ST%%'
    AND b.inference = '%s'
    '''
    
    return sql % (start_date, end_date, model_name)        
        
# 取過濾後第一片(抽測)圖片集
def get_first_wafer_filter_image(start_date, end_date, model_name):
    sql = '''
    SELECT c.LOT_END_DATE,a.ENTITY, a.RECIPE, a.LOTID, a.WAFERID, a.FILENAME, a.AI_RESULT, a.AI_CONF_N0, a.OP_RESULT, a.MATRIX_OP
    FROM AIDBA.WB_AI_IMAGE_INFO a, AIDBA.wb_ai_wafer_info b, EITEAP.AI_WAFER_INFORMATION@EAPWB c , EITEAP.AI_Wafer_detail@EAPWB d 
    WHERE (c.LOT_END_DATE between to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss') and to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss')) 
    AND c.WAFERID = d.WAFERID
    AND c.LOTID = d.LOTID
    AND c.ENTITY = d.ENTITY
    AND c.WAFERID = a.WAFERID
    AND c.LOTID = a.LOTID
    AND c.ENTITY = a.ENTITY
    AND c.WAFERID = b.WAFERID
    AND c.LOTID = b.LOTID
    AND c.ENTITY = b.ENTITY
    AND a.FILENAME = d.TIFF_FILENAME
    AND c.CUSTOMER != 'CM'
    AND (b.rework = 'N' or b.rework is null)
    AND a.DRS = 'N'
    AND b.blacklist = 'Y'
    AND c.slot = 1
    AND a.recipe not like '%%4588%%'
    AND a.recipe not like '%%FQC%%'
    AND a.recipe not like '%%IQC%%'
    AND a.recipe not like '%%BV%%'
    AND a.recipe not like '%%ST%%'
    AND b.inference = '%s'
    AND a.LOTID not in (SELECT distinct a.LOTID
    FROM AIDBA.WB_AI_IMAGE_INFO a, AIDBA.wb_ai_wafer_info b, EITEAP.AI_WAFER_INFORMATION@EAPWB c , EITEAP.AI_Wafer_detail@EAPWB d 
    WHERE (c.LOT_END_DATE between to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss') and to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss')) 
    AND c.WAFERID = d.WAFERID
    AND c.LOTID = d.LOTID
    AND c.ENTITY = d.ENTITY
    AND c.WAFERID = a.WAFERID
    AND c.LOTID = a.LOTID
    AND c.ENTITY = a.ENTITY
    AND c.WAFERID = b.WAFERID
    AND c.LOTID = b.LOTID
    AND c.ENTITY = b.ENTITY
    AND a.FILENAME = d.TIFF_FILENAME
    AND c.CUSTOMER != 'CM'
    AND (b.rework = 'N' or b.rework is null)
    AND a.DRS = 'N'
    AND b.blacklist = 'Y'
    AND c.slot != 1
    AND a.recipe not like '%%4588%%'
    AND a.recipe not like '%%FQC%%'
    AND a.recipe not like '%%IQC%%'
    AND a.recipe not like '%%BV%%'
    AND a.recipe not like '%%ST%%'
    AND b.inference = '%s')
    '''
    
    return sql % (start_date, end_date, model_name, start_date, end_date, model_name)

# 取可訓練圖片集
def get_trainable_image(start_date, end_date, model_name):
    sql = '''
    SELECT a.ENTITY, a.RECIPE, a.LOTID, a.WAFERID, a.FILENAME, a.OP_RESULT  
    FROM AIDBA.WB_AI_IMAGE_INFO a, AIDBA.wb_ai_wafer_info b, EITEAP.AI_WAFER_INFORMATION@EAPWB c , EITEAP.AI_Wafer_detail@EAPWB d 
    WHERE (c.LOT_END_DATE between to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss') and to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss')) 
    AND c.WAFERID = d.WAFERID
    AND c.LOTID = d.LOTID
    AND c.ENTITY = d.ENTITY
    AND c.WAFERID = a.WAFERID
    AND c.LOTID = a.LOTID
    AND c.ENTITY = a.ENTITY
    AND c.WAFERID = b.WAFERID
    AND c.LOTID = b.LOTID
    AND c.ENTITY = b.ENTITY
    AND a.FILENAME = d.TIFF_FILENAME
    AND c.CUSTOMER != 'CM'
    AND (b.rework = 'N' or b.rework is null)
    AND a.DRS = 'N'
    AND a.op_result is not null
    AND a.recipe not like '%%4588%%'
    AND a.recipe not like '%%FQC%%'
    AND a.recipe not like '%%IQC%%'
    AND a.recipe not like '%%BV%%'
    AND a.recipe not like '%%ST%%'
    AND b.inference = '%s'
    '''
    
    return sql % (start_date, end_date, model_name)

# 取 #2~25 N4訓練圖片集
def get_trainable_N4image(start_date, end_date, model_name):
    sql = '''
    SELECT a.ENTITY, a.RECIPE, a.LOTID, a.WAFERID, a.FILENAME, a.OP_RESULT  
    FROM AIDBA.WB_AI_IMAGE_INFO a, AIDBA.wb_ai_wafer_info b, EITEAP.AI_WAFER_INFORMATION@EAPWB c , EITEAP.AI_Wafer_detail@EAPWB d 
    WHERE (c.LOT_END_DATE between to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss') and to_date('%s 00:00:00','yyyy-mm-dd hh24:mi:ss')) 
    AND c.WAFERID = d.WAFERID
    AND c.LOTID = d.LOTID
    AND c.ENTITY = d.ENTITY
    AND c.WAFERID = a.WAFERID
    AND c.LOTID = a.LOTID
    AND c.ENTITY = a.ENTITY
    AND c.WAFERID = b.WAFERID
    AND c.LOTID = b.LOTID
    AND c.ENTITY = b.ENTITY
    AND a.FILENAME = d.TIFF_FILENAME
    AND c.CUSTOMER != 'CM'
    AND (b.rework = 'N' or b.rework is null)
    AND a.DRS = 'N'
    AND c.slot != 1
    AND a.op_result is not null
    AND a.recipe not like '%%4588%%'
    AND a.recipe not like '%%FQC%%'
    AND a.recipe not like '%%IQC%%'
    AND a.recipe not like '%%BV%%'
    AND a.recipe not like '%%ST%%'
    AND b.inference = '%s'
    '''
    
    return sql % (start_date, end_date, model_name)