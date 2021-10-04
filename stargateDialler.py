import starmap

#0. First create a starmap
A=starmap.StarMap()

#1. Add a target. Make sure to negate the components from whatever you found on the spaceship ruins in your game. 
A.addTarget(0.98644678972347,0.15794494981722,-0.044453614830485)

#2. Now add some major cartouches
# all symbols and glyphs are encoded as on page 1 and page 2 respectivly of "key.pdf".
#Example as to how a major cartouch is added. Primary symbol goes first, then all secondary symbols added clockwise from the top right position:

#
#			g6	g2 
#		 g1        d6 
#		d4	  f6    d7
#		 c7		   h7
#          g3    b7
#             b5
#
#
#			 f6*
#
# Adding the minor cartouche glyph at the end is optional. This is equivilent to the addMinorCartouche method.
A.addMajorCartouche('f6','g2','d6','d7','h7','b7','b5','h3','c7','d4','g1','g6','f6')
A.addMajorCartouche('f7','c8','b3','e3','d1','g5','d8','d3','h4','a4','b2','a1')
A.addMajorCartouche('e2','c2','a2','e7','b6','h5','h2','e3','b3','c8','a6','e1','a7')
A.addMajorCartouche('a5','c4','a3','e4','e7','a2','c2','e1','a8','h1','f3','f4','d4')
A.addMajorCartouche('d4','e5','g4','c1','b1','g6','g1','f6','b5','h3','c7','f1')
A.addMajorCartouche('a8','b8','h1','f3','a5','c2','e1','a6','c8','a1','a7','f5')
A.addMajorCartouche('h7','h6','g7','f2','b4','c5','b7','b5','f6','d6','d7','e6')

#3. Now add some minor cartouches. You can add these seperatly to adding the major cartouches.
#Symbol,Glyph
A.addMinorCartouche('f6','f6') #down
A.addMinorCartouche('e2','a7') #up
A.addMinorCartouche('a5','d4') #up 
A.addMinorCartouche('f7','c8') #down
A.addMinorCartouche('d4','a3') #up
A.addMinorCartouche('a8','d7') #up
A.addMinorCartouche('h7','d6') #down


#4. now add some starmapping data.
A.addStarmappingData('e4',-0.9818152737217,0.0,0.18983879552606)
A.addStarmappingData('f3',-0.6444904486331,0.20847820715387,-0.73564182776852)
A.addStarmappingData('h2',-0.20847820715387,0.73564182776852,0.6444904486331)
A.addStarmappingData('d4',0.20847820715387,-0.73564182776852,-0.6444904486331)
A.addStarmappingData('f6',0.18983879552606,-0.9818152737217,0.0)
A.addStarmappingData('b7',-0.39831700267993,0.85296865578697,0.3373248250886)
A.addStarmappingData('a6',-0.18983879552606,0.9818152737217,0.0)


#5. Now add coordinate logs. These are generally added in after the solver suggests what it needs to know to complete a solution.
#coordinate logs are added pretty much as they appear in the ingame record - 8 symbols in order followed by the vector numbers.
#5.1 These two are alignment logs, suggested by the solver when it needs more data.
A.addCoordinateLog('e4','g8','g8','g8','g8','g8','g8','g8',-0.85065089043125,0.0,0.52573097931216) # top alignment
A.addCoordinateLog('e4','f8','f8','f8','f8','f8','f8','f8',-0.9341724139878,0.35682194571551,8.3562715936836e-08) # bottom right alignment
A.addCoordinateLog('h4','g8','g8','g8','g8','g8','g8','g8',0.85065089043125,0.0,-0.52573097931216) # top alignment
A.addCoordinateLog('h4','e8','e8','e8','e8','e8','e8','e8',0.9341724139878,-0.35682194571551,-8.3562715936836e-08) # bottom right alignment

#5.2 These are subface mapping logs, as suggested by the solver.
A.addCoordinateLog('f7','h8','h8','h8','h8','h8','h8','h8', 0.73564182776752, 0.6444904486331, 0.20847820715387)
A.addCoordinateLog('e4','f6','e2','a5','f7','d5','a8','g7',-0.98532851142679, 0.1101689752888,0.13034769446859)
A.addCoordinateLog('e4','d5','b2','c3','b5','h3','e2','g5',-0.98753372433305, 0.14916661038742,0.050263959756707)
A.addCoordinateLog('e4','a1','b1','c1','d1','e1','f1','g1',-0.9689598085324 , 0.16938235173298 ,0.18007361930683 )
A.addCoordinateLog('e4','h1','a2','c2','d2','f2','g2','h2',-0.93383819115168, 0.0041467467210697,0.35767168917625)
A.addCoordinateLog('e4','a3','b3','d3','e3','f3','g3','h3',-0.98109339850269,-0.18471790232775, 0.057749804970055)
A.addCoordinateLog('e4','b4','c4','e4','f4','g4','h4','a5',-0.94154818386967,-0.049876784215596,0.33316561024217)
A.addCoordinateLog('e4','e5','h5','a6','b6','c6','d6','f5',-0.95070574544069,-0.302818439520290,0.066780073918481)
A.addCoordinateLog('e4','e6','g6','h6','b7','c7','d7','e7',-0.96197674711885,-0.086514234239708,0.25906760754009)
A.addCoordinateLog('e4','h7','b8','c8','d8','f4','b3','d3',-0.98985673530209, 0.06261842523099, 0.12752480699333)
A.addCoordinateLog('e4','c8','d5','f4','e4','b3','b8','e8',-0.96492370992203,-0.22203479924362, 0.14008134049597)
A.addCoordinateLog('h4','a4','c5','a7','b6','c8','f2','d7', 0.97142642482023,0.0085275108671091, -0.23718765296574)

#6. And solve.
A.solve()



