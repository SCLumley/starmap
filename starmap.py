import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull
from string import ascii_lowercase
from math import isclose
from math import trunc
from math import ceil
import itertools

class base:
	def cyclic_perm(self,a):
		n = len(a)
		b = [[a[i - j] for i in range(n)] for j in range(n)]
		return b
		
	def gridToID(self,s):
		column =  ['a','b','c','d','e','f','g','h'].index(s[0]) + 1
		row=int(s[1])-1
		out = int(row*8) + column
		if out>64:
			raise Exception("Error: Symbol grid code:" + s + "is not valid." )
#		print(s + " encodes to: " + str(out) )
		return out
		
	def IDToGrid(self,i):
		if i>64:
			raise Exception("Error: Symbol ID: " + str(i) + " is not valid." )
		column=int(i)%8-1
		row=int((i-0.1)/8) + 1
		letters=['a','b','c','d','e','f','g','h']
		l=letters[column]
		out = str(l) + str(row)
#		print(str(i) + " decodes to: " + out )
		return out
		
	def checkAddNode(self,node):
		if len(node[n]) > 2:
			return False
		return True
		
	def multidim_intersect(self,arr1, arr2):
		arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
		arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
		intersected = np.intersect1d(arr1_view, arr2_view)
		return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

	def sameFace(self,f1, f2):
		if f1 is None or f2 is None:
			return False
			
		same=False
		for p in list(itertools.permutations(f2)):
			acc=0
			acc+=abs(f1[0][0]-p[0][0])
			acc+=abs(f1[0][1]-p[0][1])
			acc+=abs(f1[0][2]-p[0][2])
			acc+=abs(f1[1][0]-p[1][0])
			acc+=abs(f1[1][1]-p[1][1])
			acc+=abs(f1[1][2]-p[1][2])
			acc+=abs(f1[2][0]-p[2][0])
			acc+=abs(f1[2][1]-p[2][1])
			acc+=abs(f1[2][2]-p[2][2])

			if isclose(acc,0) :
				same=True
		return same

	def getCentrum(self,face):
		centrumX=(face[0][0] + face[1][0] + face [2][0])/3
		centrumY=(face[0][1] + face[1][1] + face [2][1])/3
		centrumZ=(face[0][2] + face[1][2] + face [2][2])/3
		mag=(centrumX**2+centrumY**2+centrumZ**2)**0.5
		centrum=[centrumX/mag,centrumY/mag,centrumZ/mag]
		return centrum


class StarNet(base):
	def __init__(self):
		self.net = nx.Graph()
		
		for nodeID in range(1,65):
			self.net.add_node(nodeID, symGrid = self.IDToGrid(nodeID), glyphGrid=None, face=None, aligned=[False,False,False], SV=[0.0,0.0,0.0])
								
	def safeAddEdge(self,a,b):
		if self.net.has_edge(a,b) or self.net.has_edge(b,a):
			return True
		if len(list(self.net.neighbors(a))) < 3 and len(list(self.net.neighbors(b))) < 3:
			self.net.add_edge(a,b)
			return True
		else:
			raise Exception("cannot add " + self.IDToGrid(a) + " and " + self.IDToGrid(b) + " to starmap because they are an invalid connection." )
			return False
	
			
		
	
			
	def printNet(self):
		for node in self.net.nodes:
		
			print(self.net.nodes[node])
		print(self.net.edges)

	
	
	
	
	
class StarMap(base):
	def __init__(self):
		self.target=None
		self.solution=[]
		self.interpolationSolution=[]
	
		self.sn=StarNet()
		
		
		#Construct Sub-Face grid
		self.triGrid=[]
		
		for tri in range(1 , 65):

			#Address using a LIO convention
			level=ceil(tri**0.5)
			index=tri-(level-1)**2
			Orientation=index%2
			
			#Address using a half-square XYnotO convention
			NOrientation=1-Orientation
			HSY=int((index-1)/2)
			HSX=(level-1)-(HSY+NOrientation)
		
			if Orientation==1:
				t=np.array([[HSX,HSY]])
				br=np.array([[HSX,HSY+1]])
				bl=np.array([[HSX+1,HSY]])
				triFace=[t,br,bl]
		
			else:
				b=np.array([[HSX+1,HSY+1]])
				tr=np.array([[HSX,HSY+1]])
				tl=np.array([[HSX+1,HSY]])
#				triFace=[b,tr,tl] #Assume upside downs rotate
				triFace=[b,tl,tr] #Assume upside downs flip

			self.triGrid.append(triFace)

	
	
		#Construct main pentakisDodecahedron
		phi = (1 + 5 ** 0.5) / 2
		scalefactor = (3*phi+12)/19
		
		icosahedron = self.cyclic_perm([0,-1,phi]) \
              		+ self.cyclic_perm([0,1,-phi]) \
					+ self.cyclic_perm([0,-1,-phi]) \
					+ self.cyclic_perm([0,1,phi])
					
		dodecahedron =[[1,1,1]]  \
					 + self.cyclic_perm([-1,1,1]) \
					 + self.cyclic_perm([-1,-1,1]) \
					 + [[-1,-1,-1]] \
					 + self.cyclic_perm([phi,1/phi,0.0]) \
					 + self.cyclic_perm([-phi,1/phi,0.0]) \
					 + self.cyclic_perm([phi,-1/phi,0.0]) \
					 + self.cyclic_perm([-phi,-1/phi,0.0])
					 
#		print(dodecahedron)			 
		scaledIcosahedron = np.array(icosahedron)*scalefactor
		dodecahedron = np.array(dodecahedron)
#		print(scaledIcosahedron)
		
		
		
		pentakisDodecahedron=np.concatenate((scaledIcosahedron,dodecahedron),axis=0)
		
		for v in pentakisDodecahedron:
			x=v[0]
			y=v[1]
			z=v[2]
			mag=(x**2+y**2+z**2)**0.5
			v[0]=x/mag
			v[1]=y/mag
			v[2]=z/mag
		
		
		self.vertices = pentakisDodecahedron
		self.hull=ConvexHull(self.vertices)
		self.faces=[]
		centrums=[]
		for simplex in self.hull.simplices:
			p1=self.vertices[simplex[0]]
			p2=self.vertices[simplex[1]]
			p3=self.vertices[simplex[2]]
			face=[p1,p2,p3]
			self.faces.append(face)

			centrum=self.getCentrum(face)
			
			centrums.append(centrum)
		self.centrums=np.array(centrums)
		
		#add fixed symbols for subfaces:
		self.addMinorCartouche('e8','b7',False)
		self.addMinorCartouche('f8','h8',False)
		self.addMinorCartouche('g8','a1',False)
		self.addMinorCartouche('h8','g4',False)
		
		
	def addMajorCartouche(self,a,b,c,d,e,f,g,h,i,j,k,l,gl=None):
		print("adding major cartouche")
		aID=self.gridToID(a)
		bID=self.gridToID(b)
		cID=self.gridToID(c)
		dID=self.gridToID(d)
		eID=self.gridToID(e)
		fID=self.gridToID(f)
		gID=self.gridToID(g)
		hID=self.gridToID(h)
		iID=self.gridToID(i)
		jID=self.gridToID(j)
		kID=self.gridToID(k)
		lID=self.gridToID(l)

		#Add edges from central node
		self.sn.safeAddEdge(aID,cID)
		self.sn.safeAddEdge(aID,gID)
		self.sn.safeAddEdge(aID,kID)

		#Add edges from 3 nearest neigbours		
		self.sn.safeAddEdge(bID,cID)
		self.sn.safeAddEdge(cID,dID)
		self.sn.safeAddEdge(fID,gID)
		self.sn.safeAddEdge(gID,hID)
		self.sn.safeAddEdge(jID,kID)
		self.sn.safeAddEdge(kID,lID)
			
		#Add remaining Edges
		self.sn.safeAddEdge(lID,bID)
		self.sn.safeAddEdge(dID,eID)
		self.sn.safeAddEdge(eID,fID)
		self.sn.safeAddEdge(hID,iID)
		self.sn.safeAddEdge(iID,jID)
		
		if gl != None:
			self.addMinorCartouche(a,gl,False)
		self.updateMap()
		
		
	def addMinorCartouche(self,s,g,verbose=True,ignore=False,force=False):	
		if verbose:
			print("adding minor cartouche")
			
		
		newGlyphPair=(s,g)
		sID=self.gridToID(s)
		gID=self.gridToID(g)
		oldGlyphPair=None

		
		for i in self.sn.net.nodes:
			if self.sn.net.nodes[i]['glyphGrid'] == g:
				oldGlyphPair=(self.sn.net.nodes[i]['symGrid'],self.sn.net.nodes[i]['glyphGrid'])
				break
			if self.sn.net.nodes[i]['symGrid'] == s and self.sn.net.nodes[i]['glyphGrid'] is not None:
				oldGlyphPair=(self.sn.net.nodes[i]['symGrid'],self.sn.net.nodes[i]['glyphGrid'])
				break
				
		
		if oldGlyphPair is not None and oldGlyphPair != newGlyphPair:
			if ignore:
				print("Warning: Trying to map symbol-glyph pair" + str(newGlyphPair) + "but it is already mapped to " + str(oldGlyphPair) + ". Please check your coordinate logs are correct")
				if not force:
					return
			else:
				raise Exception("Error: Trying to map symbol-glyph pair" + str(newGlyphPair) + "but it is already mapped to " + str(oldGlyphPair) + ". Please check your coordinate logs, and your minor cartouche entrries are correct")
		
		if oldGlyphPair is None:
			if verbose:
				print("adding glyph")
				print(sID,gID)
			self.sn.net.nodes[sID]['glyphGrid']=g
			self.sn.net.nodes[sID]['subFace']=self.triGrid[gID-1]
			return
		
		if oldGlyphPair == newGlyphPair:
			if verbose:
				print("This glyph has already been added for this symbol.")
			return
		


	
	
	
	def addStarmappingData(self,symbol,SVx,SVy,SVz):
		print("Adding Starmapping Data")
		sID=self.gridToID(symbol)
		i=0
		bestdp=0
		bestface=0
		match=False
		preexsisting=False
		for centrum in self.centrums:
			if self.sn.net.nodes[sID]['face'] != None and not isclose(SVx,centrum[0]) and not isclose(SVy,centrum[1]) and not isclose(SVz,centrum[2]):
				oldSV=self.sn.net.nodes[sID]['SV'] 
				if not isclose(SVx,oldSV[0]) or not isclose(SVy,oldSV[1]) or not isclose(SVz,oldSV[2]):
					raise Exception("Starmapping data Error - Symbol: " + symbol + " has already been mapped to SV: " + str(self.sn.net.nodes[sID]['SV']) +  " and can't be mapped again to: " + str([SVx,SVy,SVz]))
					break
				else:
					print("Starmapping data already matches with a mapped symbol: " + symbol + ". Nothing new to add here.")
					return
					
			if isclose(SVx,centrum[0]) and isclose(SVy,centrum[1]) and isclose(SVz,centrum[2]) :
				self.sn.net.nodes[sID]['face']=self.faces[i]
				self.sn.net.nodes[sID]['SV']=[SVx,SVy,SVz]
				bestface=i
				match=True
				break
			i+=1
		
		if match :
			print("Starmapping Data successfully added")
			self.updateMap()
		else:
			raise Exception("Starmapping data Error - Symbol: " + symbol + " does not fit on the star map. Please check Coordinates." )
		
		
		
	def updateMap(self,layers=0):
		#Goes through each node, and checks if any two of the neighboring nodes have mapped positional data. 
		print("updating starmap with recursion depth: " + str(layers))
		recurse=False
		for nodeID in self.sn.net.nodes:
			neighbors=self.sn.net.neighbors(nodeID)
			mappedNNs=[]
			unmappedNNs=[]
			for neighbor in neighbors:
				if self.sn.net.nodes[neighbor]['face'] != None:
					mappedNNs.append(self.sn.net.nodes[neighbor]['face'])
				else:
					unmappedNNs.append(self.sn.net.nodes[neighbor]['symGrid'])


			if len(mappedNNs) == 2: 
				print("Sufficent data to extend map around node: " + str(nodeID))
				if not self.sn.net.nodes[nodeID]['face']: # If THIS node is not mapped to a face, there is now enough datat to map THIS node.	
					print("Mapping this node.")
					i=0
					for face in self.faces: # The two mapped neighbors will share a point. The face to map will share two points with one neighbor and two points with the other.
						if len(self.multidim_intersect(np.array(mappedNNs[0]),np.array(face)))==2 and len(self.multidim_intersect(np.array(mappedNNs[1]),np.array(face)))==2:
							#Once found, map the starmap face to the node, and add centrum data.
							print("Position of face: " + str(i) + " maps to symbol " + self.sn.net.nodes[nodeID]['symGrid'])
							self.sn.net.nodes[nodeID]['face']=face
							self.sn.net.nodes[nodeID]['SV']=[self.centrums[i][0],self.centrums[i][1],self.centrums[i][2]]
							recurse=True
							break
						i+=1
				else: # If this node IS mapped to a face, there is now enough data to map the remaining node.
					print("Mapping adjacent node.")
					i=0
					for face in self.faces: #the one unmapped face will be the only one to share two points with this face, that isn't already mapped. 
						if len(self.multidim_intersect(np.array(self.sn.net.nodes[nodeID]['face']),np.array(face)))==2 and len(unmappedNNs) == 1 and  not ( self.sameFace(face,mappedNNs[0]) or self.sameFace(face,mappedNNs[1]) ):
							print("Position of face: " + str(i) + " maps to symbol " + unmappedNNs[0])
							self.sn.net.nodes[self.gridToID(unmappedNNs[0])]['face']=face
							self.sn.net.nodes[self.gridToID(unmappedNNs[0])]['SV']=[self.centrums[i][0],self.centrums[i][1],self.centrums[i][2]]
							recurse=True
							break
						i+=1
					
					
		if(recurse):
			layers+=1
			self.updateMap(layers)
		else:
			print("Insufficent data to extend map.")

		
	def addTarget(self,SVx,SVy,SVz):
		self.target=np.array([[SVx,SVy,SVz]])
		pass
	

	
	def addCoordinateLog(self,g1,g2,g3,g4,g5,g6,g7,g8,SVx,SVy,SVz):
		vector=np.array([[SVx,SVy,SVz]])
		
		print("Adding coordinate log")
		#First, special case if it is an address designed to get a constellation coordinate:
		if g2 == 'h8' and g3 == 'h8' and g4 == 'h8' and g5 == 'h8' and g6 == 'h8' and g7 == 'h8' and g8 == 'h8':
			print("This address maps symbol: " + g1 + " to it's appropriate face")
			self.addStarmappingData(g1,SVx,SVy,SVz)
			return
			
			
		#Second, Special cases to align a face (if needed) to order [T,BR,BL]
		alignmentVertex=None
		if g2 == 'g8' and g3 == 'g8' and g4 == 'g8' and g5 == 'g8' and g6 == 'g8' and g7 == 'g8' and g8 == 'g8': #Align to top
			vertexString="Top"
			alignmentVertex=0
		if g2 == 'f8' and g3 == 'f8' and g4 == 'f8' and g5 == 'f8' and g6 == 'f8' and g7 == 'f8' and g8 == 'f8': #Align to Bottom Right
			vertexString="Bottom Right"
			alignmentVertex=1 
		if g2 == 'e8' and g3 == 'e8' and g4 == 'e8' and g5 == 'e8' and g6 == 'e8' and g7 == 'e8' and g8 == 'e8': #Align to Bottom left
			vertexString="Bottom left"
			alignmentVertex=2			
		if alignmentVertex is not None:
			print("Aligning face for symbol " + g1 + " to " + vertexString)	
			g1ID=self.gridToID(g1)
			alignment=self.sn.net.nodes[g1ID]['aligned']
			
			targetFace=None
			allFaces=nx.get_node_attributes(self.sn.net,'face')
			for i in allFaces:
				if self.sn.net.nodes[i]['symGrid'] == g1 :
					targetFace=self.sn.net.nodes[i]['face']
			if targetFace == None:
				print("Warning: There is no face mapped to symbol: " + g1 + ". It is not recommended to try and perform a face alignment on a face that has not yet been mapped, due to numerical precision errors. Instead, please map the face by either using adding starmapping data, or by inputing the coordinate log for the following address, before this one:")
				print(g1,"h8","h8","h8","h8","h8","h8","h8")
				print("I will continue, but any solution produced is likley to be wrong.")
				targetFace=self.getTargetFace(vector)
				centrum=self.getCentrum(targetFace)
				self.addStarmappingData(g1,centrum[0],centrum[1],centrum[2])

			self.alignFace(g1ID,vector,alignmentVertex)
			return
		
		#Finally try to use coordinate logs to map out which sub-face is which
		targetFace=self.getTargetFace(vector)
		allFaces=nx.get_node_attributes(self.sn.net,'face')
		facesymbol=None
		for i in allFaces:
			if self.sameFace(allFaces[i],targetFace):
				facesymbol=self.sn.net.nodes[i]['symGrid']
				targetFace=self.sn.net.nodes[i]['face'] # neccisary to get aligned face from net, rather than unalighned face from pentakis dodecahedron
				targetFaceID=i
				break
		
		if facesymbol==None:
			centrum=self.getCentrum(targetFace)
			self.addStarmappingData(g1,centrum[0],centrum[1],centrum[2])
			
		faceAligned=self.sn.net.nodes[targetFaceID]['aligned']
		if sum(faceAligned)<2:
			print("Info: First symbol mapped, but on an unaligned face. There is no point attempting to use this face to determine the symbol arrangment on the subface. Please align this face first:")
			print("Dial any two of the following addresses with the stargate, and add them as coordinate data.")
			print(g1,"e8","e8","e8","e8","e8","e8","e8")
			print(g1,"f8","f8","f8","f8","f8","f8","f8")
			print(g1,"g8","g8","g8","g8","g8","g8","g8")
			return
		
		#face is aligned. Try to add subface data
		subfaceSymbols=[g2,g3,g4,g5,g6,g7,g8]
		glyphSol=self.getGlyphsol(vector,targetFace,True)
		
		print("adding glyph data from coordinate log")
		for i,sym in enumerate(subfaceSymbols):
			symID=self.gridToID(sym)
			self.addMinorCartouche(sym,glyphSol[i])
		
	def solve(self):
		
		print("Attempting Solution")
		if self.target is None:
			print("Info: A target vector is unspecified. Please specify one to find a solution.")
			exit(0)
		print("attempting to find solution for vector: ",self.target)
		
		#First, Find face on the main starmap.
		targetFace=self.getTargetFace(self.target)
		targetFaceID=None
					
		#Now see if it matches a symbol in the starnet.
		allFaces=nx.get_node_attributes(self.sn.net,'face')
		for i in allFaces:
			if self.sameFace(allFaces[i],targetFace):
				self.solution.append(self.sn.net.nodes[i]['symGrid'])
				targetFace=self.sn.net.nodes[i]['face']
				targetFaceID=i
				break
		
		if len(self.solution)==0:
			print("Info: Cannot find solution, the starmap is not complete enough to decode the first symbol")
			self.suggest()
			return
			
			
			
		faceAligned=self.sn.net.nodes[targetFaceID]['aligned']
		if sum(faceAligned)<2:
			print("Info: First simbol found, but on an unaligned face. Solution not possible without more data.")
			self.suggest()
			return
		
		
		glyphSol=self.getGlyphsol(self.target,targetFace)
		interSol=self.getGlyphsol(self.target,targetFace,"interpol")
		
		print("glyph Solution is: ", glyphSol)
		
		
		for i,g in enumerate(glyphSol):
			symbol=g + "*"
			for n in self.sn.net.nodes:
				if self.sn.net.nodes[n]['glyphGrid'] == g:
					symbol=self.IDToGrid(n)
					break
			self.solution.append(symbol)
			self.interpolationSolution.append(interSol[i])
			
					
					
		incomplete= any('*' in string for string in self.solution)

		if incomplete:
			self.suggest()
		else:
			print("Solution Found! Code is:")
			print(self.solution)
	
		
	def suggest(self):
	
		if len(self.solution)==0:
			print("To get closer to a solution, try the following:")
			print("1. Find and add more major Cartouche data. Add data for NEW cartouches, especially if they contain the following symbols:")
			for n in self.sn.net.nodes:
				if self.sn.net.nodes[n]['face'] != None:
					print(self.IDToGrid(n))
			print("2. Add more starmapping data. The following symbols are still needed:")
			for n in self.sn.net.nodes:
				if self.sn.net.nodes[n]['face'] == None:
					print(self.IDToGrid(n))
			print("3. Dial some of the following addresses with the stargate, and add them as coordinate data.")
			for n in self.sn.net.nodes:
				neighbors=self.sn.net.neighbors(n)
				nn=0
				for neighbor in neighbors:
					nn+=1
				if nn > 1 and self.sn.net.nodes[n]['face'] == None:
					print(self.IDToGrid(n),"h8","h8","h8","h8","h8","h8","h8")
					
		if len(self.solution)==1:
			print("To get closer to a solution, try the following:")
			print("1. Dial any two of the following addresses with the stargate, and add them as coordinate data.")
			print(self.solution[0],"e8","e8","e8","e8","e8","e8","e8")
			print(self.solution[0],"f8","f8","f8","f8","f8","f8","f8")
			print(self.solution[0],"g8","g8","g8","g8","g8","g8","g8")
		
		
		if len(self.solution)>1:
			gl=0
			for i in self.sn.net.nodes:
				if self.sn.net.nodes[i]['glyphGrid']== None:
					gl+=1
			print("An incomplete solution has been found: ",self.solution)
			print("To get closer to a solution, try the following:")
			print("1. Add more minor cartouches")
			print("There are currently " + str(gl) + " more glyphs to map.")
			print("2. Dial a the following addresses with the stargate, and add them as coordinate data:")
			i=0
			address=[self.solution[0]]
			if gl >= 7:
				for nodeID in self.sn.net.nodes:
					if self.sn.net.nodes[nodeID]['glyphGrid'] == None:
						address.append(self.sn.net.nodes[nodeID]['symGrid'])
					if len(address)==8:
						print(address)
						address=[self.solution[0]]
						i+=8
			else:
				for nodeID in self.sn.net.nodes:
					if self.sn.net.nodes[nodeID]['glyphGrid'] == None:
						address.append(self.sn.net.nodes[nodeID]['symGrid'])
				while(len(address)<8):
					address.append('any')
				print(address)
				
					

				
			print("It is possible you will not need all of them, but the more you dial, the more likley the solution will be found.")
				

		pass			
	
	#Takes in a face and constructs a basis for it
	def getFaceBasis(self,f,rightHanded=True):
		top=f[0]
		bottomRight=f[1]
		bottomLeft=f[2]
		x=np.subtract(bottomLeft,top)
		y=np.subtract(bottomRight,top)
		if rightHanded:
			return [ x , y ]
		else:
			return [ y , x ]
		
	#Takes in a vector, and a face. The vector is projected through the face. Returns a coordinate in terms of the basis of the face.
	#Basis is determined using first point of the face in array as the origin. 
	def interpolate(self,v,f,rightHanded=True):
		top=f[0]
		fBasis = [ x , y ] = self.getFaceBasis(f,rightHanded)
		N=np.cross(x,y)
		
		if np.dot(v,N) != 0:
			t=np.dot(top,N)/np.dot(v,N)
			vSubTop=np.subtract(v*t,top)

			#From a linear algebra perspective, these really should work, but i'm going with a hand-coded approach because i've tested it more reliably
			#invfBasis=np.linalg.pinv(fBasis)
			#vectorInPlanePlaneBasis=np.dot(vectorInPlaneStandardBasis,invfBasis)

			interpolationQuotient=fBasis[0][1]*fBasis[1][0] - fBasis[0][0]*fBasis[1][1]
			interpX,interpY,zTest = 0.0,0.0,1.1
			if interpolationQuotient==0:
				return []
			else:
				interpX = ( fBasis[1][0]*vSubTop[0][1]-fBasis[1][1]*vSubTop[0][0] ) / interpolationQuotient
				interpY = ( fBasis[0][1]*vSubTop[0][0]-fBasis[0][0]*vSubTop[0][1] ) / interpolationQuotient
				zTest=interpX*fBasis[0][2] + interpY*fBasis[1][2] - vSubTop[0][2]

			if not isclose(zTest,0,abs_tol=1e-09):
				raise Exception("Error: Something has gone wrong with the face interpolation. The zTest goves a value of " + str(zTest) + " When it should be near zero.")
			
			vectorInPlanePlaneBasis = np.array([[interpX,interpY]])
			return vectorInPlanePlaneBasis
		else:
			return []
			
	def frac_to_oct(self,f, n=8):
		# store the number before the decimal point
		whole = int(f)
		rem = (f - whole) * 8
		int_ = int(rem)
		rem = (rem - int_) * 8
		octals = [str(int_)]
		count = 1
		# loop until 8 * rem gives you a whole num or n times
		while rem and count < n:
			count += 1
			int_ = int(rem)
			rem = (rem - int_) * 8
			octals.append(str(int_))
		return float("{:o}.{}".format(whole, "".join(octals)))




		
		
	def getTargetFace(self,vector):
		bestface=[0,None]
		for face in self.faces:
			centrum = self.getCentrum(face)
			dp=np.dot(vector,centrum)
			if dp > bestface[0]:
				bestface=[dp,face]

		targetFace=bestface[1]
		return targetFace
		
	def getGlyphsol(self,vector,face,output="glyph",noflip=True):

		gridScale=8
		
		glyphSol=[]
		idSol=[]
		triSol=[]
		subFaceSol=[]
		interSol=[]
		
		subFace=face
		
		loop=0
		
		while(len(glyphSol)<7):
			interpol=self.interpolate(vector,subFace)
			interpol=interpol*gridScale

			interpolTrunc = [[
								trunc(interpol[0][0]),
								trunc(interpol[0][1])
							]]
			interpolRem=[[ 
							interpol[0][0]-interpolTrunc[0][0],
						    interpol[0][1]-interpolTrunc[0][1]
						]]

			triCoords = (
							interpolTrunc[0][0],
							interpolTrunc[0][1],
							trunc(interpolRem[0][0]+interpolRem[0][1])
						)

			#Use tricoords to calculate outputs
			sum=triCoords[0]+triCoords[1]+triCoords[2]
			id=sum**2 + 2*triCoords[1] + triCoords[2] + 1
			glyphSol.append(self.IDToGrid(id))
			idSol.append(id)
			triSol.append(triCoords)
			triFace=self.triGrid[id-1]
			subFaceSol.append(subFace)
			interSol.append(interpol)
			
			#Now calculate the newface from the subface
			fbasis = np.array(self.getFaceBasis(subFace))
			newFace=[]
			for i,point in enumerate(triFace):
				# np = T + x*fi + y*fj
				a=subFace[0][0] + (point[0][0]/gridScale)*fbasis[0][0] + (point[0][1]/gridScale)*fbasis[1][0]
				b=subFace[0][1] + (point[0][0]/gridScale)*fbasis[0][1] + (point[0][1]/gridScale)*fbasis[1][1]
				c=subFace[0][2] + (point[0][0]/gridScale)*fbasis[0][2] + (point[0][1]/gridScale)*fbasis[1][2]
				newPoint=np.array([a,b,c])

				newFace.append(newPoint)

			subFace=newFace
				
			loop+=1

		#Returns
		if output == "subFace":
			return subFaceSol
		if output == "id":
			return idSol
		if output == "tri":
			return triSol
		if output == "interpol":
			return interSol
		return glyphSol
		
		
	def alignFace(self,faceID,vector,vertex):
		face=self.sn.net.nodes[faceID]['face']		
		aligned=self.sn.net.nodes[faceID]['aligned']
	
		distances=[
			np.linalg.norm(face[0]-np.array(vector)),
			np.linalg.norm(face[1]-np.array(vector)),
			np.linalg.norm(face[2]-np.array(vector))
		]
		
		indexPoint = distances.index(min(distances))
	
	
	
		if sum(aligned)==0:
			n=vertex-indexPoint
			face=self.rotateTri(face,n)
			aligned[vertex]=True
			self.sn.net.nodes[faceID]['face']=face
			self.sn.net.nodes[faceID]['aligned']=aligned
			return
		
		if sum(aligned)==1:
			if vertex-indexPoint == 0:
				aligned=[True,True,True]
				self.sn.net.nodes[faceID]['aligned']=aligned
				return
			else:
				fix=aligned.index(True)
				face=self.flipTri(face,fix)
				aligned=[True,True,True]
				self.sn.net.nodes[faceID]['face']=face
				self.sn.net.nodes[faceID]['aligned']=aligned
		
		if sum(aligned)>=2:
			return		
		
	def rotateTri(self,face,n):
		n=n%3
		if n==0:
			return face
		if n>0:
			face=[face[2],face[0],face[1]]
			return self.rotateTri(face,n-1)
		if n<0:
			face=[face[1],face[2],face[0]]
			return self.rotateTri(face,n+1)
	
	
	def flipTri(self,face,n):
		n=n%3
		if n==0:
			return [face[0],face[2],face[1]]
		if n==1:
			return [face[2],face[1],face[0]]
		if n==2:
			return [face[1],face[0],face[2]]
		
		

		
		
####################### UNUSED STUFF

#This was an attempt at me being clever using a base-8 conversion of the interpolation to directly map to the triangles. I see no reason why it shouldn't work, but it was a pain to debug, so I went with the simpler approach. 
	def getGlyphsol2(self,vector,face,output="glyph"):
		#Now interpolate using a base 8 coordinate system
		q=7
		interpolation=self.interpolate(vector,face)
		octal=[ format(self.frac_to_oct(interpolation[0][0]), '.8f') , format(self.frac_to_oct(interpolation[0][1]), '.8f')  ]

		print(interpolation)
		print(octal)
		#form each pair of digits in the octal interpolation into a triagonal coordinate
		zipCoords=list(map(list,zip(octal[0],octal[1])))
		zipCoords=zipCoords[2:]
		print(zipCoords)
		
		triCords=[]
		flip=False
		for i,[u,v] in enumerate(zipCoords): #iterate over each pair
			downface=False
			if i < len(zipCoords)-1:
				for j in range(1,len(zipCoords)): #look at remaining pairs to see if we will overflow into downface
					[nu,nv]=zipCoords[i+j]
					print("current",[u,v],"next",[nu,nv])
					if flip:
						[tx,ty]=[q-int(u),q-int(v)]
						[tnx,tny]=[q-int(nu),q-int(nv)]
					else:
						[tx,ty]=[int(u),int(v)]
						[tnx,tny]=[int(nu),int(nv)]
						
					print("[tx,ty]",[tx,ty])
					if tnx+tny  > q :
						downface=True
						break
					if tnx+tny  < q :
						break

					print("partial triCoord",[tx,ty])
					print("looping")
				
				#upface or downface
				if downface : #downface
					triCord=[tx,ty,1]
					flip=not flip
					print("down")
					pass
					
				else: # upface
					triCord=[tx,ty,0]
					print("up")
					pass
				triCords.append(triCord)
		print(triCords)
		
		#convert triagonal coordinates into a glyph symbol
		glyphSol=[]
		idSol=[]
		for i,(x,y,u) in enumerate(triCords):
			print(x,y,u)
			sum=x+y+u
			id=sum**2 + 2*y + u + 1
			print("i",id)
			glyphSol.append(self.IDToGrid(id))
			idSol.append(id)
		if output == "glyph":
			return glyphSol
		if output == "id":
			return idSol
		if output == "tris":
			return triCords



		
		
		


