import socket
import sys
import struct
import time
import tcp
import threading
import _thread
import pymysql


def mysqlconnect():
    connection = pymysql.connect(
        host='8.137.159.75',
        port=3307,
        user='',
        password='',
        database='ks',
        charset='utf8'
    )
    return connection


def readdata():   # 连接数据库
    conn = mysqlconnect()
    cursor = conn.cursor()
    cursor.execute('select * from test')
    aa = cursor.fetchall()
    print(aa)
    #cursor.execute(sql, [num, yb, wd, time])
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':

    try:
        # MySQLConnect()
        readdata()
        # ReadData()
        print("连接成功")
    except:
        print("连接失败")
        sys.exit(1)

