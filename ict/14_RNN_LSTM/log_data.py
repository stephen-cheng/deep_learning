import InsertTools

def load_data():
	conn=InsertTools.getConnection()
	cur=conn.cursor()
	sql="select timestamp, logid, severity from log_s1;"
	cur.execute(sql)
	res=cur.fetchall()
	sql2="select timestamp, logid, severity from log_p1;"
	cur.execute(sql2)
	res2=cur.fetchall()
	severity=['fatal','failure','critical','error','warning','notice','info','NULL']
	se_index=[0,1,2,3,4,5,6,7]
	train,test,train_str,test_str,se_raw,se_raw2,se_no,se_no2=[],[],[],[],[],[],[],[]
	for r in res:
		train.append(r[1])
		train_str.append(str(r[1]))
	for r in res2:
		test.append(r[1])
		test_str.append(str(r[1]))
	for r in res:
		se_raw.append(r[2])
	for sr in se_raw:
		number=se_index[severity.index(sr)]
		se_no.append([number])
	for r2 in res2:
		se_raw2.append(r2[2])
	for sr2 in se_raw2:
		number2=se_index[severity.index(sr2)]
		se_no2.append([number2])
	cur.close()
	conn.commit()
	conn.close
	return train,test,train_str,test_str
	
if __name__=='__main__':
	datatrain, datatest, datatrain_str, datatest_str = load_data()
	dataset = ' '.join(datatrain_str[:100])
	print len(dataset)