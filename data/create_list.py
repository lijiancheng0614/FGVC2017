import json
import os

# calc the id for a taskId of class
# arch_Label 
arch_Label = {"pants":[], "outerwear":[], "shoe":[], "dress":[]}
taskId_Class = {}
with open('fgvc4_iMat.task_map.json', 'r') as f:
	data = json.load(f)
	taskinfo = data["taskInfo"]

	for item in taskinfo:
		taskname = item["taskName"]
		taskid = item["taskId"]

		bigclass, subclass = taskname.split(":")
		arch_Label[bigclass].append([subclass, taskid, 0])

		if taskid in taskId_Class.keys():
			taskId_Class[taskid].append(bigclass)
		else:
			taskId_Class[taskid] = []
			taskId_Class[taskid].append(bigclass)

	arch_Label["pants"].sort()
	arch_Label["outerwear"].sort()
	arch_Label["shoe"].sort()
	arch_Label["dress"].sort()

	print(taskId_Class)

for i in range(len(arch_Label["pants"])):
	arch_Label["pants"][i][2] = i

for i in range(len(arch_Label["outerwear"])):
	arch_Label["outerwear"][i][2] = i

for i in range(len(arch_Label["shoe"])):
	arch_Label["shoe"][i][2] = i

for i in range(len(arch_Label["dress"])):
	arch_Label["dress"][i][2] = i

num_pants = len(arch_Label["pants"])
num_outerwear = len(arch_Label["outerwear"])
num_shoe = len(arch_Label["shoe"])
num_dress = len(arch_Label["dress"])

print("pants:\n", arch_Label["pants"], num_pants)
print("outerwear:\n", arch_Label["outerwear"], num_outerwear)
print("shoe:\n", arch_Label["shoe"], num_shoe)
print("dress:\n", arch_Label["dress"], num_dress)

# calc the id for a label of taskId
taskId_Label = {}
with open('fgvc4_iMat.train.data.json', 'r') as f:
	data = json.load(f)
	annotations = data["annotations"]

	for item in annotations:
		if item["taskId"] in taskId_Label.keys():
			taskId_Label[item["taskId"]].add(item["labelId"])
		else:
			taskId_Label[item["taskId"]] = set()
			taskId_Label[item["taskId"]].add(item["labelId"])

labelId_labelName = {}
with open('fgvc4_iMat.label_map.json', 'r') as f:
	data = json.load(f)
	labelInfo = data["labelInfo"]

	for item in labelInfo:
		labelName = item["labelName"]
		labelId = item["labelId"]
		labelId_labelName[labelId] = labelName

for k in taskId_Label:
	# k is taskId
	#print(k)

	taskId_Label[k] = list(taskId_Label[k])
	taskId_Label[k].sort()
	#taskId_Label[k]  is  all label of task k
	#print(taskId_Label[k])

	#print("taskid:", k, "range:0-",len(taskId_Label[k])-1)
	label_name = [labelId_labelName[name] for name in taskId_Label[k]]
	print("taskid:", k)
	print(label_name)
	temp ={}
	for i in range(len(taskId_Label[k])):
		temp[taskId_Label[k][i]] = i

	#change label to 0, 1, 2, ..
	taskId_Label[k] = temp
	#print(k)
	#print(taskId_Label[k])

img_id = {}

with open('fgvc4_iMat.train.data.json', 'r') as f:
	data = json.load(f)
	annotations = data["annotations"]
	for item in annotations:
		# a item["taskId"] only crospending one class
		id_class = taskId_Class[item["taskId"]][0]

		all_task = arch_Label[id_class]
		for task in all_task:
			if task[1] == item["taskId"]:
				id_task = task[2]

		id_label = taskId_Label[item["taskId"]][item["labelId"]]

		if item["imageId"] in img_id.keys():
			img_id[item["imageId"]].append([id_class, id_task, id_label])
		else:
			img_id[item["imageId"]] = []
			img_id[item["imageId"]].append([id_class, id_task, id_label])

with open('pants_list.txt', 'w') as p_f:
	with open('outerwear_list.txt', 'w') as o_f:
			with open('shoe_list.txt', 'w') as s_f:
				with open('dress_list.txt', 'w') as d_f:
					for v in img_id:
						if not os.path.isfile('train_images/'+v+".jpg"):
							print('train_images/'+v+".jpg"+"is not exist")
							continue

						p_b = False
						o_b = False
						s_b = False
						d_b = False

						p_s = [v+".jpg"]+[" -1"]*num_pants
						o_s = [v+".jpg"]+[" -1"]*num_outerwear
						s_s = [v+".jpg"]+[" -1"]*num_shoe
						d_s = [v+".jpg"]+[" -1"]*num_dress

						#if v == "292":
							#print(img_id[v])
						for item in img_id[v]:
							#print(bigClass)
							#print(item)
							if item[0] == "pants":
								p_b = True
								p_s[item[1]+1] = " {}".format(item[2])
							elif item[0] == "outerwear":
								o_b = True
								o_s[item[1]+1] = " {}".format(item[2])
							elif item[0] == "shoe":
								s_b = True
								s_s[item[1]+1] = " {}".format(item[2])
							elif item[0] == "dress":
								d_b = True
								d_s[item[1]+1] = " {}".format(item[2])

						if p_b:
							p_f.write(''.join(p_s)+"\n")
						if o_b:
							o_f.write(''.join(o_s)+"\n")
						if s_b:
							s_f.write(''.join(s_s)+"\n")
						if d_b:
							d_f.write(''.join(d_s)+"\n")

val_img_id = {}

with open('fgvc4_iMat.validation.data.json', 'r') as f:
	data = json.load(f)
	annotations = data["annotations"]
	for item in annotations:
		# a item["taskId"] only crospending one class
		id_class = taskId_Class[item["taskId"]][0]

		all_task = arch_Label[id_class]
		for task in all_task:
			if task[1] == item["taskId"]:
				id_task = task[2]

		id_label = taskId_Label[item["taskId"]][item["labelId"]]

		if item["imageId"] in val_img_id.keys():
			val_img_id[item["imageId"]].append([id_class, id_task, id_label])
		else:
			val_img_id[item["imageId"]] = []
			val_img_id[item["imageId"]].append([id_class, id_task, id_label])

with open('val_pants_list.txt', 'w') as p_f:
	with open('val_outerwear_list.txt', 'w') as o_f:
			with open('val_shoe_list.txt', 'w') as s_f:
				with open('val_dress_list.txt', 'w') as d_f:
					for v in val_img_id:
						if not os.path.isfile('val_images/'+v+".jpg"):
							print('val_images/'+v+".jpg"+"is not exist")
							continue

						p_b = False
						o_b = False
						s_b = False
						d_b = False

						p_s = [v+".jpg"]+[" -1"]*num_pants
						o_s = [v+".jpg"]+[" -1"]*num_outerwear
						s_s = [v+".jpg"]+[" -1"]*num_shoe
						d_s = [v+".jpg"]+[" -1"]*num_dress

						#if v == "292":
							#print(img_id[v])
						for item in val_img_id[v]:
							#print(bigClass)
							#print(item)
							if item[0] == "pants":
								p_b = True
								p_s[item[1]+1] = " {}".format(item[2])
							elif item[0] == "outerwear":
								o_b = True
								o_s[item[1]+1] = " {}".format(item[2])
							elif item[0] == "shoe":
								s_b = True
								s_s[item[1]+1] = " {}".format(item[2])
							elif item[0] == "dress":
								d_b = True
								d_s[item[1]+1] = " {}".format(item[2])

						if p_b:
							p_f.write(''.join(p_s)+"\n")
						if o_b:
							o_f.write(''.join(o_s)+"\n")
						if s_b:
							s_f.write(''.join(s_s)+"\n")
						if d_b:
							d_f.write(''.join(d_s)+"\n")

