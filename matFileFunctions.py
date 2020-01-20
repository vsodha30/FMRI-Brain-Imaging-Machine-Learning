def getInfoTrial(matFile, trialNumber):
	"""
	info.mint gives the time of the first image in the interval (the minimum time)
	info.maxt gives the time of the last image in the interval (the maximum time)
	info.cond has possible values 0,1,2,3.  
		Cond=0: ignored. 
		Cond=1: rest, 
		Cond=2 sentence/picture is not negated.  
		Cond=3 sentence/picture is negated.  
	info.firstStimulus: is either 'P' or 'S' 
	info.sentence gives the sentence presented during this trial.
	info.img describes the image presented during this trial.  
	info.actionAnswer: has values -1 or 0.  
		0: expected to press the answer button
		-1: not to press the answer
	info.actionRT: gives the reaction time of the subject
		Time is in milliseconds.
	"""
	info = {}
	info['cond'] = matFile['info'][0][trialNumber]['cond']
	info['len'] = matFile['info'][0][trialNumber]['len']
	info['mint'] = matFile['info'][0][trialNumber]['mint']
	info['maxt'] = matFile['info'][0][trialNumber]['maxt']
	info['firstStimulus'] = matFile['info'][0][trialNumber]['firstStimulus']
	info['sentence'] = matFile['info'][0][trialNumber]['sentence']
	info['sentenceRel'] = matFile['info'][0][trialNumber]['sentenceRel']
	info['sentenceSym1'] = matFile['info'][0][trialNumber]['sentenceSym1']
	info['sentenceSym2'] = matFile['info'][0][trialNumber]['sentenceSym2']
	info['img'] = matFile['info'][0][trialNumber]['img']
	info['actionAnswer'] = matFile['info'][0][trialNumber]['actionAnswer']
	info['actionRT'] = matFile['info'][0][trialNumber]['actionRT']
	return info
	
def getDataTrial(matFile, trialNumber):
	"""
	data contains n X 4698 matrix, n = len of maxt-mint 
	"""
	data = matFile['data'][trialNumber][0]
	return data

def getMeta(matFile):
	"""
	meta.study 
		name of the fMRI study
	meta.subject
		identifier for the human subject
	meta.ntrials 
		number of trials in this dataset
	meta.nsnapshots 
		total number of images in the dataset
	meta.nvoxels
		number of voxels (3D pixels) in each image
	meta.dimx
		maximum x coordinate in the brain image.   
	meta.dimy 
		maximum y coordinate in the brain image.   
	meta.dimz
		maximum z coordinate in the brain image.   
	meta.colToCoord(v,:) 
		gives the geometric coordinate (x,y,z) of the voxel
		corresponding to column v in the data
	meta.coordToCol(x,y,z) 
		gives the column index (within the data) of the voxel
		whose coordinate is (x,y,z)
	meta.rois 
		struct of form (roi.name, roi.coord([x,y,z]), roi.column)
		roi.coord is list of coordinates included in that roi
	meta.colToROI{v} 
		gives the ROI of the voxel corresponding to column v in the
		data.  
	"""
	meta = {}
	meta['study'] = matFile['meta'][0][0]['study'][0]
	meta['subject'] = matFile['meta'][0][0]['subject'][0]
	meta['ntrials'] = matFile['meta'][0][0]['ntrials'][0]
	meta['nsnapshots'] = matFile['meta'][0][0]['nsnapshots'][0]
	meta['nvoxels'] = matFile['meta'][0][0]['nvoxels'][0]
	meta['dimx'] = matFile['meta'][0][0]['dimx'][0]
	meta['dimy'] = matFile['meta'][0][0]['dimy'][0]
	meta['dimz'] = matFile['meta'][0][0]['dimz'][0]
	
	meta['colToCoord'] = []
	for _ in matFile['meta'][0][0]['colToCoord']:
		meta['colToCoord'].append(_)
	
	meta['rois'] = {}
	for [x,y,z] in matFile['meta'][0][0]['rois'][0]:
		meta['rois'][x[0]] = {}
		meta['rois'][x[0]]['coords'] = y
		meta['rois'][x[0]]['columns'] = z[0]
		
	meta['colToROI'] = matFile['meta'][0][0]['colToROI'][0]
	return meta

def matFileToDataWithImportantColumns(matFile, importantColumns):
	meta = getMeta(matFile)
	## now we prepare the dataset, we will only include the important columns
	numberOfTrials = meta['ntrials'][0]
	DATA = []
	CLASS = []
	for i in range(numberOfTrials):
		infoTrial = getInfoTrial(matFile, i)
		##############  CONSIDERING THE THIRD CLASS 'rest': 1  #################
		############# according to our classification, frame[1,8] = firstStimulus
		############                                   frame[9:17] = rest
		############                                   frame[18:26] = ~firstStimulus
		############                 P:-1 | S:1
		if(infoTrial['cond'] in [2,3]):
			lenData = infoTrial['len']
			dataTrial = getDataTrial(matFile, i)
			""" first 8 image frames for firstStimulus """
			for j in dataTrial[:8]:
				DATA.append(list(map(j.__getitem__, importantColumns)))
				CLASS.append(-1 if infoTrial['firstStimulus']=='P' else 1)
			"""for j in dataTrial[8:16]:
				DATA.append(list(map(j.__getitem__, importantColumns)))
				CLASS.append(0)"""
			for j in dataTrial[16:26]:
				DATA.append(list(map(j.__getitem__, importantColumns)))
				CLASS.append(1 if infoTrial['firstStimulus']=='S' else -1)
	return [DATA, CLASS]

def singleImportantColumn(matFile, columnName):
	meta = getMeta(matFile)
	## now we prepare the dataset, we will only include the important columns
	numberOfTrials = meta['ntrials'][0]
	DATA = []
	CLASS = []
	for i in range(numberOfTrials):
		infoTrial = getInfoTrial(matFile, i)
		if(infoTrial['cond'] in [2,3]):
			lenData = infoTrial['len']
			dataTrial = getDataTrial(matFile, i)
			for j in dataTrial[:16]:
				DATA.append(list(map(j.__getitem__, columnName)))
				CLASS.append(1 if infoTrial['firstStimulus']=='P' else -1)
			for j in dataTrial[16:26]:
				DATA.append(list(map(j.__getitem__, columnName)))
				CLASS.append(-1 if infoTrial['firstStimulus']=='S' else 1)
	return [DATA, CLASS]

	
def matFileToData(matFile):
	meta = getMeta(matFile)
	## now we prepare the dataset, we will only include the important columns
	numberOfTrials = meta['ntrials'][0]
	DATA = []
	CLASS = []
	for i in range(numberOfTrials):
		infoTrial = getInfoTrial(matFile, i)
		##############  CONSIDERING THE THIRD CLASS 'rest': 1  #################
		############# according to our classification, frame[1,16] = firstStimulus
		############                                   frame[16:32] = ~firstStimulus
		############                 P:-1 | S:1                  
		if(infoTrial['cond'] in [2,3]):
			lenData = infoTrial['len']
			dataTrial = getDataTrial(matFile, i)
			for j in dataTrial[:16]:
				DATA.append(j)
				CLASS.append(-1 if infoTrial['firstStimulus']=='P' else 1)
			for j in dataTrial[16:]:
				DATA.append(j)
				CLASS.append(1 if infoTrial['firstStimulus']=='S' else -1)
	return [DATA, CLASS]
	
def getRegionOfInterest(matFile):
	return matFile['meta'][0]['roi'][0][0].split("_")