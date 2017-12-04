f = open('F:\\SHA256SUM')
l = f.readlines()
f.close()
fName = open('F:\\task_events_file_name.txt', 'w')
for i in l:
    if i.count('task_events')>0:
        fileAddr = i.split()[1][1:]
        fileName = fileAddr.split('/')[1]
        fName.write(fileName+'\r\n')
fName.close()