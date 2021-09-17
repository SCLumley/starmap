import starmap


#First create a starmap
A=starmap.StarMap()

#Now add a target
A.addTarget(-0.98644678972347,-0.15794494981722,0.044453614830485)

#Now add some major cartouches
# all symbols and glyphs are encoded as on page 1 and page 2 respectivly of "key.pdf".
#Example as to how a major cartouch is added. Primary symbol goes first, then all secondary symbols added clockwise from the top right position:
A.addMajorCartouche('f6','g2','d6','d7','h7','b7','b5','g3','c7','d4','g1','g6','f6')
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

#Other cartouches I have
A.addMajorCartouche('e2','c2','a2','e7','b6','g5','h2','e3','b3','c8','a6','e1','a7')
A.addMajorCartouche('a5','c4','a3','e4','e7','a2','c2','e1','a8','h1','f3','f4','d4')


#Now add some minor cartouches. You can add these separatly to adding the major cartouches.
A.addMinorCartouche('f6','f6')
A.addMinorCartouche('e2','a7')
A.addMinorCartouche('a5','d4')
A.addMinorCartouche('f7','c8')
A.addMinorCartouche('d5','a3')
A.addMinorCartouche('a8','d7')
A.addMinorCartouche('g7','d6')


#now add some starmapping data.
A.addStarmappingData('e4',-0.9818152737217,0.0,0.18983879552606)
A.addStarmappingData('f6',0.18983879552606,-0.9818152737217,0.0)
A.addStarmappingData('f3',-0.6444904486331,0.20847820715387,-0.73564182776852)
A.addStarmappingData('d4',0.20847820715387,-0.73564182776852,-0.6444904486331)

#now add coordinate logs.
#These two are alignment logs, suggested by the solver when it needs more data.
A.addCoordinateLog('e4','f8','f8','f8','f8','f8','f8','f8',-0.9341724139878,0.35682194571551,8.3562715936836e-08)
A.addCoordinateLog('e4','g8','g8','g8','g8','g8','g8','g8',-0.85065089043125,0.0,0.52573097931216)

#These are subface mapping logs, as suggested by the solver.
A.addCoordinateLog('e4','d5','b2','c3','b5','h3','e2','g5',-0.98753372433305,0.14916661038742,0.050263959756707)
A.addCoordinateLog('e4','a1','b1','c1','d1','e1','f1','g1',-0.9689598085324 ,0.16938235173298 ,0.18007361930683 )
A.addCoordinateLog('e4','h1','a2','c2','d2','f2','g2','h2',-0.93383819115168,0.0041467467210697,0.35767168917625)
A.addCoordinateLog('e4','a3','b3','d3','e3','f3','g3','a4',-0.98109339850269,-0.18471790232775,0.057749804970055)
A.addCoordinateLog('e4','b4','c4','e4','f4','g4','h4','c5',-0.94154818386967,-0.049876784215596,0.33316561024217)
A.addCoordinateLog('e4','e5','h5','a6','b6','c6','d6','f5',-0.95070574544069,-0.302818439520290,0.066780073918481)
A.addCoordinateLog('e4','e6','g6','h6','b7','c7','d7','e7',-0.96197674711885,-0.086514234239708,0.25906760754009) #check this one
A.addCoordinateLog('e4','h7','b8','c8','d8','c7','b3','d3',-0.98985673530209, 0.06261842523099, 0.12752480699333)


#And solve.
A.solve()