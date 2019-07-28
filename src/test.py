import pymysql  # 导入 pymysql
import json
import codecs

# 打开数据库连接
'''
db= pymysql.connect(host="39.106.6.6",user="haiou",
     password="haiou",db="haiou",port=3306,cursorclass=pymysql.cursors.DictCursor)
'''
db = pymysql.connect(host="39.106.6.6", user="loushuai",
                     password="loushuai", db="CompanyInfo", port=3306, cursorclass=pymysql.cursors.DictCursor)

# 使用cursor()方法获取操作游标
cur = db.cursor()

# 1.查询操作
# 编写sql 查询语句  user 对应我的表名
# sql = "select * from data"
sql = "alter table data add rate varchar(2) null"
sql2 = "select * from company_base_info limit 1,100"
# f_res = codecs.open('rate.txt', 'w', 'utf-8')
try:
    #  cur.execute(sql)  #执行sql语句
    cur.execute(sql2)
    results = cur.fetchall()  # 获取查询的所有记录
    cnt = 0
    #  score_set = []
    for item in results:
        print(item)
except Exception as e:
    raise e