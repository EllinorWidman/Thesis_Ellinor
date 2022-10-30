import os

rootdir = 'my_root_dir/FakeNewsNet/code/fakenewsnet_dataset'
result_root_dir = 'my_root_dir/StyleBased/Resultat'

def get_news_files():
	print(rootdir)
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
			newFileName = os.path.join(subdir, file).replace("/", "_").replace("_mnt_c_users_EllinorWidman_FakeNewsNet_code_","").replace("_news content.json","")
			if newFileName.find('real') >= 0:
				resultdir = result_root_dir + "/Real"
			elif newFileName.find('fake') >= 0:
				resultdir = result_root_dir +  "/Fake"
			newFileNameWithPath = os.path.join(resultdir, newFileName+".txt")
			if os.path.exists(newFileNameWithPath):
				os.remove(newFileNameWithPath)
			newTestFile = open(newFileNameWithPath, "x")
			f = open (os.path.join(subdir, file), "r")
			textContent = f.read()
			posText = textContent.find('"text":')
			#print(posText)
			posImg = textContent.find('"images":')
			#print(posImg)
			posTitle = textContent.find('"title":')
			#print(posTitle)
			posMeta = textContent.find('"meta_data":')
			#print(posMeta)
			textContent = (textContent[posTitle:posMeta])+ (textContent[posText:posImg])
			newTestFile.write(str(textContent))

			newTestFile.close()

