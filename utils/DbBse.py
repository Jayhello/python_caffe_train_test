# _*_ coding:utf-8 _*_
import MySQLdb


class DbBase(object):
    def __init__(self, **kwargs):
        self.db_config_file = kwargs['db_config']
        self.config_db(self.db_config_file)

    def config_db(self, db_config):
        data = db_config
        host = data['host']
        user = data['user']
        pwd = data['pwd']
        db = data['db']
        port = data['port']
        self.conn = MySQLdb.connect(host=host, port=port, user=user, passwd=pwd, db=db, charset="utf8", use_unicode=True)
        self.cursor = self.conn.cursor()


class DbService(DbBase):
    def __init__(self, **kwargs):
        super(DbService, self).__init__(**kwargs)

    def get_ad_info(self):
        """
        return all id and url
        [(1, 'http://xxx.1.jpg'), (2, 'http://xxx.2.jpg).....]
        :return:
        """
        sql = """select id, url from ad_text_web_set_2017818"""
        self.cursor.execute(sql)

        return [row for row in self.cursor]

    def update_ad_info(self, lst):
        """
        write predict result to database
        :param lst:
        :return:
        """
        try:
            sql = """
                    update ad_text_web_set_2017818
                    set label_txt=%s, label_web=%s, label_others=%s,modify_date=now()
                    where id=%s
                  """ % (lst[0], lst[1], lst[2], lst[3])
            self.cursor.execute(sql)
            self.conn.commit()
        except Exception, e:
            print e


def get_default_db():
    ip = '127.0.0.1'
    port = 3307
    user = 'user'
    pwd = 'user'
    db = 'caffe'

    db_config = {}
    db_config['host'] = ip
    db_config['port'] = port
    db_config['user'] = user
    db_config['pwd'] = pwd
    db_config['db'] = db

    return DbService(db_config=db_config)


if __name__ == '__main__':

    pass
