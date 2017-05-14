import string
import xml.etree.ElementTree as ET
import MySQLdb
class InsertHelper(object):
    __hash__ = object.__hash__
    __class__ = 'InsertHelper'

    def __init__(self):
        self.tableName = None
        self.startedEvent = None
        self.finishedEvent = None
        self.failedEvent = None
        self.tags = None
        self.cols = None
        self.params = None
        self.isSucc = False
        self.isCompleted = False
        self.uniqueTag = None
        self.uniqueTagValue = None
        self.typeTag = None
        self.path = None
        self.keysFilter = None

    def paramsIsCompleted(self):
        if self.params is None: return False
        assert isinstance(self.params, dict)
        l = filter(lambda k: self.params[k] is None, self.params.keys())
        return len(l) == 0

    def getKeysAndValuesAsString(self):
        assert isinstance(self.params, dict)
        keys = filter(lambda k: self.params[k] is not None, self.params.keys())
        values = [self.params[k] for k in keys]
        keysString = ",".join(keys)
        valuesString = ""
        for i in range(len(values) - 1):
            if isinstance(values[i], unicode):
                valuesString += "'" + values[i] + "',"
            else:
                valuesString += str(values[i]) + ","
        if isinstance(values[-1], unicode):
            valuesString += "'" + values[-1] + "'"
        else:
            valuesString += str(values[-1])
        return keysString, valuesString

    def col2tag(self,colName):
        return self.tags[self.cols.index(colName)]

    def getInsertSql(self):
        sql = string.Template("""insert into $tableName($keys) values($values)""")
        vars = {"tableName": self.tableName}
        keysString, valuesString = self.getKeysAndValuesAsString()
        vars["keys"] = keysString
        vars["values"] = valuesString
        return sql.substitute(vars)
		
    def find(self,obj,key):
        if obj[self.typeTag] not in self.path[key][self.typeTag]:return None
        pathid=self.path[key][self.typeTag].index(obj[self.typeTag])
        assert isinstance(self.path[key],dict)
        res=obj
        for location in self.path[key]["path"][pathid]:
            res=res[location]
        return res

    def greedyFill(self, obj):
        assert isinstance(self.params, dict)
        for k in self.uniqueTag:
            if self.uniqueTagValue[self.uniqueTag.index(k)] is None:
                self.uniqueTagValue[self.uniqueTag.index(k)]=self.find(obj,k)
        emptyCols = filter(lambda k: self.params[k] is None, self.params.keys())
        for k in emptyCols:
            self.params[k]=self.find(obj,k)

    def isStarted(self, obj):
        assert isinstance(obj, dict)
        if not obj.__contains__(self.typeTag): return False
        return obj[self.typeTag] in self.startedEvent

    def isFinished(self, obj):
        assert isinstance(obj, dict)
        if not obj.__contains__(self.typeTag): return False
        return obj[self.typeTag] in self.finishedEvent

    def isFailed(self, obj):
        assert isinstance(obj, dict)
        if not obj.__contains__(self.typeTag): return False
        return obj[self.typeTag] in self.failedEvent

    def isMatch(self, obj):
        for k in self.uniqueTag:
            if self.uniqueTagValue[self.uniqueTag.index(k)] != self.find(obj,k):
                return False
        return True
        pass

    def fill(self, obj):
        if self.isStarted(obj):
            self.greedyFill(obj)
        elif self.isFinished(obj):
            if self.isMatch(obj):
                self.greedyFill(obj)
                self.isSucc = True
                self.isCompleted = True
        elif self.isFailed(obj):
            if self.isMatch(obj):
                self.greedyFill(obj)
                self.isSucc = False
                self.isCompleted = True
        else:
            pass
        return self

    def canGetKeys(self,parent):
        assert isinstance(parent,InsertHelper)
        # for key in self.keysFilter:
        #     if key not in parent.params.keys():
        #         return False
        #     if key not in self.params.keys():
        #         self.params[key]=parent.params[key]
        #         return True
        # return False
        for key in self.keysFilter:
            if key not in parent.params.keys():
                return False
            else:
                self.params[key]=parent.params[key]
        return True
        pass

    def isParent(self):
        pass
def getConnection():
    et=ET.parse("dbconfig.xml")
    root=et.getroot()
    conn=MySQLdb.connect(
        host=root.findtext("host"),
        port=int(root.findtext("port")),
        user=root.findtext("user"),
        passwd=root.findtext("password"),
        db=root.findtext("db")
    )
    return conn
