import urllib2

url = 'https://commondatastorage.googleapis.com/clusterdata-2011-1/'
f = open('F:\\SHA256SUM')
l = f.readlines()
f.close()
for i in l:
	if i.count('task_usage') > 0:
	#if i.count('task_constraints') > 0:
	#if i.count('machine_events') > 0:
	#if i.count('machine_attributes') > 0:
	#if i.count('job_events') > 0:
	#if i.count('task_events')>0:
		fileAddr = i.split()[1][1:]
		fileName = fileAddr.split('/')[1]
		print 'downloading', fileName
		data = urllib2.urlopen(url+fileAddr).read()
		print 'saving', fileName
		fileDown = open('F:\\task_usage\\'+fileName, 'wb')
		fileDown.write(data)
		fileDown.close()