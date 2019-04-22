from pymongo import MongoClient
import time
import csv


class Database:

    def __init__(self):
        self.mongoCluster = 'mongodb+srv://admin:<password>@cluster0-oqumv.mongodb.net/test?retryWrites=true'
        self.mongodbUser = None
        self.mongodbUserPwd = None
        self.mongoCl = False
        self.mongoDb = None
        self.mongoCol = None
        self.set_userpwd()
        self.connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        if exc_type:
            raise exc_type(exc_val)

        return self

    def set_userpwd(self):
        with open('userpwd.txt', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.mongodbUser = row[0]
                self.mongodbUserPwd = row[1]

    def connect(self):
        if not self.mongoCl:
            self.mongoCl = MongoClient(self.mongoCluster, username=self.mongodbUser, password=self.mongodbUserPwd)

    def disconnect(self):
        if self.mongoCl:
            self.mongoCl.close()

    def db(self, mdb):
        self.mongoDb = self.mongoCl[mdb]

    def collection(self, mcol):
        self.mongoCol = self.mongoDb[mcol]

    def read(self, col, req):
        pass

    def write(self, col, req):
        pass

    def insert_one(self, mdb, mcol, mpayload):
        self.db(mdb)
        self.collection(mcol)
        mpayload['createdAt'] = time.time()
        _id = self.mongoCol.insert_one(mpayload).inserted_id
        return _id

    def update_one(self, mdb, mcol, mfilter, mpayload):
        self.db(mdb)
        self.collection(mcol)
        self.mongoCol.update_one(mfilter, mpayload)

