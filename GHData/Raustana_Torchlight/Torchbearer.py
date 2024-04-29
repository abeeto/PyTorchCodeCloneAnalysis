import MalmoPython
import os
import random
import sys
import time
import json
import random
import math
import errno
import itertools # This is rather important for combinatorial work.
from timeit import default_timer as timer

trial6x6 = 	'''
	<DrawingDecorator>
		<DrawCuboid x1="0" y1="40" z1="0" x2="15" y2="40" z2="15" type="air"/>
		<DrawCuboid x1="0" y1="39" z1="0" x2="5" y2="39" z2="5" type="obsidian"/>
		<DrawCuboid x1="0" y1="39" z1="0" x2="15" y2="39" z2="15" type="air"/>
	</DrawingDecorator>
	'''

trial8x8 = 	'''
	<DrawingDecorator>
		<DrawCuboid x1="0" y1="40" z1="0" x2="15" y2="40" z2="15" type="air"/>
		<DrawCuboid x1="0" y1="39" z1="0" x2="7" y2="39" z2="7" type="obsidian"/>
		<DrawCuboid x1="0" y1="39" z1="0" x2="15" y2="39" z2="15" type="air"/>
	</DrawingDecorator>
	'''

trial10x10 = 	'''
	<DrawingDecorator>
		<DrawCuboid x1="0" y1="40" z1="0" x2="15" y2="40" z2="15" type="air"/>
		<DrawCuboid x1="0" y1="39" z1="0" x2="9" y2="39" z2="9" type="obsidian"/>
		<DrawCuboid x1="0" y1="39" z1="0" x2="15" y2="39" z2="15" type="air"/>
	</DrawingDecorator>
	'''

def CreateTrial( n , startingCoordinates):
	# Creates an nXn square. Erases ground level and one above; rewrites with obsidian.
	"""
	Args:
				n:						<int>	How big the square is.
				startingCoordinates:	<tuple(int, int)>	Where the square begins. This is important if we are to make multi-grid tests.
	"""

	return '''
	<DrawingDecorator>
		<DrawCuboid x1="''' + str(startingCoordinates[0]) + '''" y1="40" z1="''' + str(startingCoordinates[1]) + '''" x2="''' + str(n - 1) + '''" y2="40" z2="''' + str(n - 1) + '''" type="air"/>
		<DrawCuboid x1="''' + str(startingCoordinates[0]) + '''" y1="39" z1="''' + str(startingCoordinates[1]) + '''" x2="''' + str(n - 1) + '''" y2="39" z2="''' + str(n - 1) + '''" type="air"/>
		<DrawCuboid x1="''' + str(startingCoordinates[0]) + '''" y1="39" z1="''' + str(startingCoordinates[1]) + '''" x2="''' + str(n - 1) + '''" y2="39" z2="''' + str(n - 1) + '''" type="obsidian"/>
	</DrawingDecorator>
	'''

def GetMissionXML( trial ):
	# generatorString = 2;0;127; for MC v1.7 and below
	return '''<?xml version="1.0" encoding="UTF-8" ?>
	<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
		<About>
			<Summary>Light the way!</Summary>
		</About>

		<ServerSection>
			<ServerInitialConditions>
				<AllowSpawning>false</AllowSpawning>
				<Time>
					<StartTime>14000</StartTime>
					<AllowPassageOfTime>false</AllowPassageOfTime>
				</Time>
				<Weather>clear</Weather>
			</ServerInitialConditions>
			<ServerHandlers>
				<FlatWorldGenerator generatorString="3;0;127;"/>
				''' + trial + '''
				<ServerQuitFromTimeUp timeLimitMs="45000"/>
				<ServerQuitWhenAnyAgentFinishes />
			</ServerHandlers>
		</ServerSection>

		<AgentSection mode="Creative">
			<Name>Lightbringer</Name>
			<AgentStart>
				<Placement x="0" y="40" z="0"/>
				<Inventory>
					<InventoryItem slot="0" type="torch"/>
				</Inventory>
			</AgentStart>
			<AgentHandlers>
			<ContinuousMovementCommands turnSpeedDegs="360"/>
			<MissionQuitCommands quitDescription="quit"/>
			</AgentHandlers>
		</AgentSection>

	</Mission>'''


class Torchbearer(object):
	def __init__(self, trialsize):
		"""
		Create Torchbearer AI, with empty lists of coordinates.

		Args:
			trialsize:	<int>	The size of the square that the AI will iterate over, as an nXn square.
		"""

		# Doable: the minimum amount of torches needed to fill up a num by num space.
		# Because torches light up 14 on their own square and taxi cab downwards...
		# ... if we start in the center of a 7x7 matrix:

		#	8	9	10	11
		#	9	10	11	12
		#	10	11	12	13
		#	11	12	13	14	13	12	11
		#				13	12
		#				12		10
		#				11			8

		# The same goes for a 6x6 matrix onwards.
		# If you make an 8x8 or 9x9 matrix, you require 2 torches.
		# If you make a 10x10 or 11x11 matrix, you require 3.
		# And so on.

		# Essentially: take your nXn matrix, divide by 2, round down, subtract 2, and that's how much your doable is.

		self.doable = int(math.floor(trialsize/2)-2)

		# NOTE: A majority of this code is deprecated, but remains as a back up if our new code fails.
		# See "V 1.0" at the bottom.

		self.currentList = [] # What our current light levels are at.
		self.startList = [] # What coordinates we start with.

		for i in range(trialsize):
			self.currentList.append([0]*trialsize)
			for j in range(trialsize):
				self.startList.append((i,j))
				
		self.trial = trialsize

		self.startList = self.doableList(self.doable, self.startList) # Create a list of coordinates to do. DOES NOT WORK WITH V 1.0.

		self.triedList = []
		self.bestList = []

		self.position = (0,0)

		self.currentTorches = []
		self.worst = []

		# Create a dict of scores --
		self.scoredList = dict()

		#currentList = list of light levels of the CURRENT mission
		#triedList = list of coordinates we have TRIED already, as a list of combinations/final (x,z) coordinates
			#(or: [ [(tuple), (tuple), (tuple)], [(tuple), (tuple), (tuple)] ])
		#bestList = list of BEST coordinates to place torches, as a list of (x,z) coordinates
			#(or: [(tuple), (tuple), (tuple)])
			#(chosen compared to bList, based on number of torches placed/len of the combination)

		#currentTorches = list of tuples in the current run.
		#worst = the longest list of tuples in triedList so far.

		# scoredList: a dictionary of "scores" and their coordinates. The score is calculated based on how many squares are lit up.

	def compareToCenter(self, a,b, center):
		# First, br
		if(a[0] < center and b[0] < center and a[1] < center and b[1] < center):
			return True
		# Now, tr
		elif(a[0] < center and b[0] < center and a[1] > center and b[1] > center):
			return True
		# Now, bl
		elif(a[0] > center and b[0] > center and a[1] < center and b[1] < center):
			return True
		# Now, tl
		elif(a[0] > center and b[0] > center and a[1] > center and b[1] > center):
			return True
		else:
			return False

	def compare(self, target):
		# Ranked the girds, given the a low index to the grids that are close to center, a high index to outside grids.
		# Usage sort(list, key = lambda x : compare(x))
		center = self.findCenter(self.trial, self.trial)
		return max(abs(target[0] - center[0]), abs(target[1] - center[1]))
		
	def doableList(self, doable, startingList):
		# Creates a list of all combinations in len(startingList) C doable (as an nCr function)

		# TODO: Start at the center and move outwards.

		# for i in center:
			# if i[0] < ((self.trial-1)/2):
				# if i[1] < ((self.trial-1)/2):
					# check.append((i[0]-1,i[1]))
					# check.append((i[0],i[1]-1))
				# else:
					# check.append((i[0]-1,i[1]))
					# check.append((i[0],i[1]+1))
			# else:
				# if i[1] < ((self.trial-1/2)):
					# check.append((i[0]+1,i[1]))
					# check.append((i[0],i[1]-1))
				# else:
					# check.append((i[0]+1,i[1]))
					# check.append((i[0],i[1]+1))

		retList = []
        # <<<< Raustana
		# for i in itertools.combinations(startingList,doable):
		# 	if(sorted(list(i)) not in retList):
		# 		retList.append(sorted(list(i)))
        # <<<< Ruidong
        # Reduce the time space from O(N^2) to O(N)
        # compare function is the algorithm we used in updateList function
		for i in itertools.combinations(sorted(startingList, key = lambda x : self.compare(x)), doable):
			if i not in retList:
				retList.append(i)
		return retList

	def updateLists(self):
		# Updates the set of lists by placing a torch at the CURRENT POSITION.
		initialNum = 14 # Torch light level = 0
		for i,j in enumerate(self.currentList): # i = index of y; j = list at y
			disY = abs(self.position[1] - i)
			for a,b in enumerate(j): # a = index of x; b = number at x
				disX = abs(self.position[0] - a)
				tryNum = initialNum - (disX + disY) # Taxi Cab distance
				if(b > 14): # WALL IMPLEMENTATION
					break
				elif(b < tryNum): # The coordinate is closer than 14 squares away, but is darker than it would be if a torch was placed down.
					j[a] = tryNum
		return

	def placeTorch(self, agent_host):
		# Places torch down in game world; does not affect algorithm.
		agent_host.sendCommand("use 1")
		time.sleep(0.1)
		agent_host.sendCommand("use 0")
		self.updateLists()


	def findCenter(self, x, z):
		# Finds center of trial
		retX = math.floor(x/2)
		retZ = math.floor(z/2)
		return (int(retX), int(retZ))

	def teleport(self, agent_host, teleport_x, teleport_z):
		"""Directly teleport to a specific position."""
		print ("Attempting teleport to",str(teleport_x),str(teleport_z))
		tp_command = "tp " + str(teleport_x)+ " 40 " + str(teleport_z)
		agent_host.sendCommand(tp_command)
		good_frame = False
		start = timer()
		while not good_frame:
			world_state = agent_host.getWorldState()
			if not world_state.is_mission_running:
				print ("Mission ended prematurely - error.")
				exit(1)
			if not good_frame and world_state.number_of_video_frames_since_last_state > 0:
				frame_x = world_state.video_frames[-1].xPos
				frame_z = world_state.video_frames[-1].zPos
				if math.fabs(frame_x - teleport_x) < 0.001 and math.fabs(frame_z - teleport_z) < 0.001:
					good_frame = True
					end_frame = timer()
		self.position = (teleport_x, teleport_z)

	def clearSelf(self):
		self.currentList = []
		for i in range(self.trial):
			self.currentList.append([0]*self.trial)
		self.currentTorches = []

# Create default Malmo objects:
if __name__ == '__main__':
	# For consistent results -- UNCOMMENT
	# random.seed(0)
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

	my_client_pool = MalmoPython.ClientPool()
	my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))

	agent_host = MalmoPython.AgentHost()
	try:
		agent_host.parse( sys.argv )
	except RuntimeError as e:
		print 'ERROR:',e
		print agent_host.getUsage()
		exit(1)
	if agent_host.receivedArgument("help"):
		print agent_host.getUsage()
		exit(0)

	num = 8
	#
	#
	#Allowed preset trials: trial6x6, trial8x8, trial10x10
	trial = CreateTrial(num, (0,0))
	#
	#

	if(trial == trial6x6):
		num = 6
	elif(trial == trial8x8):
		num = 8
	elif(trial == trial10x10):
		num = 10

	if(num > 7):
		num_reps = min(num**2, 500) + 1
	else:
		num_reps = 10

	# Initialize torchbearer with trial size (num)
	torchbearer = Torchbearer(num)
	print("Trial size: {} x {}".format(num, num))
	# Initialize breaking point, where there's "too many torches"
	breaker = 10

	for iRepeat in range(num_reps):

		my_mission = MalmoPython.MissionSpec(GetMissionXML(trial), True)
		my_mission_record = MalmoPython.MissionRecordSpec()
		my_mission.allowAllAbsoluteMovementCommands()
		my_mission.requestVideo(800, 500)
		my_mission.setViewpoint(1)
		my_mission.startAtWithPitchAndYaw(0, 40, 0, 0, 90)


		# Attempt to start a mission:
		max_retries = 3
		for retry in range(max_retries):
			try:
				agent_host.startMission( my_mission, my_mission_record )
				break
			except RuntimeError as e:
				if retry == max_retries - 1:
					print "Error starting mission:",e
					exit(1)
				else:
					time.sleep(2)

		# Loop until mission starts:
		print "Waiting for the mission to start ",
		world_state = agent_host.getWorldState()
		while not world_state.has_mission_begun:
			sys.stdout.write(".")
			time.sleep(0.1)
			world_state = agent_host.getWorldState()
			for error in world_state.errors:
				print "Error:",error.text

		print
		print "Mission running # ",iRepeat+1

		world_state = agent_host.getWorldState()
		for error in world_state.errors:
			print "Error:",error.text

		center = torchbearer.findCenter(num, num)
		# print("Are we running?",world_state.is_mission_running)

		agent_host.sendCommand("pitch 1")
		# print("Trying to look down")
		time.sleep(0.25)

		# Control center torch
		# torchbearer.teleport(agent_host, center[0], center[1])
		# torchbearer.placeTorch(agent_host)

		# If this is not the "victory lap":
		if(iRepeat != num_reps - 1):

			# Check Torchbearer for list of light levels; quit if all are 8 or lower.
			if(len(torchbearer.startList) > 0):
				tryThis = torchbearer.startList[0] # tryThis is a list of 2-tuples/coordinates.
				for a in tryThis:	# Each a is a coordinate.
					torchbearer.teleport(agent_host,a[0],a[1])
					torchbearer.placeTorch(agent_host)
				# print(torchbearer.currentList)
				dark = 0
				for x in torchbearer.currentList: # For each list in the list of lists...
					# print(x)
					for y in x: # For each number in the list...
						# print(y)
						if(y < 8):
							dark += 1
				# print(dark)
				if(dark in torchbearer.scoredList):
					torchbearer.scoredList[dark].append(tryThis)
				else:
					torchbearer.scoredList[dark] = [tryThis]
				# print(torchbearer.scoredList)
				torchbearer.startList.remove(torchbearer.startList[0])
				torchbearer.clearSelf()

			#															#
			#			V 1.0, without doablelist implemented!			#
			#															#

			'''
			# Start at this location first; technically iterating over the whole list, slowly.
			if(len(torchbearer.startList) > 0):
				coordinate = torchbearer.startList[0]
				torchbearer.teleport(agent_host, coordinate[0], coordinate[1])
				torchbearer.placeTorch(agent_host)
				torchbearer.startList.remove(coordinate)




			# The current run's torchbearing.
			while True:
				dark = False;
				for i in torchbearer.currentList:
					for j in i:
						if(j < 8): # 8 is light level that will respawn
							dark = True

				# SUCCESS!
				if(not dark):
					endList = []
					for i,j in enumerate(torchbearer.currentList): # i is index, j is list
						for a,b in enumerate(j): # a is index, b is number
							if(b == 14):
								endList.append((a, i))
					#
					# TODO: check if the list is already tried.
					# If it isn't, we should add it.
					if(sorted(endList) not in torchbearer.triedList):
						torchbearer.triedList.append(sorted(endList))
						print("The area is alight!")
					else:
						print("Already tried this solution.")
					# print(torchbearer.currentList)
					#
					# Check if this tried but valid list is the longest so far
					# Really only works on the initial run.
					if(len(endList) > len(torchbearer.worst)):
						torchbearer.worst = endList
					torchbearer.clearSelf()
					break

				# DARK SPACES!
				# Currently, we place a torch on the squares with the LOWEST light level.
				# We need to expand it so it checks ALL squares, to find the best solutions.
				# For example: the 8x8 trial has the best solutions in a set of pairs, where each coordinate is adjacent to the center squares but diagonally opposite to the other:
				#		0	0
				#	T	C	C	0
				#	0	C	C	T
				#		0	0
				# This fills the entire space with light, in the least amount of torches.
				# As was given to us, though, the whole problem is O(n^3), and that gets out of hand SUPER fast.
				else:
					# Iterate twice: once to check for the amount of lowest light levels, the other time to add them.
					darkList = []
					lowest = 14
					for i in torchbearer.currentList:
						for j in i:
							if(j < lowest):
								lowest = j
					for i,j in enumerate(torchbearer.currentList): # i is index, j is list
						for a,b in enumerate(j): # a is index, b is number
							if(b == lowest):
								darkList.append((a,i))
				# print(darkList)

				# COMPLETELY RANDOM DISTRIBUTION
				# NO SMART CHECKS
				# NEED ALGORITHM

				# Check list of dark (light < 8) squares; check if this string of coordinates has already turned out a solution.
				# If it has, do not continue.
				tempDarkList = []
				tempDarkList.extend(darkList)
				for i in tempDarkList:
					tryTheseTorches = []
					tryTheseTorches.extend(torchbearer.currentTorches)
					tryTheseTorches.append(i)
					if(sorted(tryTheseTorches) in torchbearer.triedList):
						darkList.remove(i)

				if(len(darkList) == 0):
					print("Already tried this solution.")
					torchbearer.clearSelf()
					break

				rando = random.randint(0, len(darkList) - 1)

				torchbearer.teleport(agent_host, darkList[rando][0], darkList[rando][1])
				torchbearer.placeTorch(agent_host)
				torchbearer.currentTorches.append(torchbearer.position)

				# Check if the placed torches are either above our suspected control break, or more torches than our worst solution at the very start.
				# If we end up getting lucky with a 1 torch solution, then we never have to check any higher.
				# Because of how we iterate initially (off startList), that will rarely happen...
				# Instead, we get the "baseline" -- if we start at a corner, like most people are inclined to do, what will the "worst" be?
				if(len(torchbearer.currentTorches) > breaker or (len(torchbearer.worst) > 0 and len(torchbearer.currentTorches) > len(torchbearer.worst))):
					print("Inefficient choices.")
					torchbearer.clearSelf()
					break

			#DEBUG BREAK
			#break
			# print(torchbearer.triedList)
			'''

			#															#
			#						END V 1.0							#
			#															#


			agent_host.sendCommand("quit")
			time.sleep(0.1)

		else:
			lowest = None

			for i in torchbearer.scoredList: # First iterate to find the lowest score in torchbearer.scoredList and if it contains coordinates.
				if((lowest == None or i < lowest) and len(torchbearer.scoredList[i]) > 0):
					lowest = i
			for i in torchbearer.scoredList[lowest]: # Then iterate over the list of scoredList at the lowest score.
				torchbearer.bestList.append(i)
			print("Best Coordinates for grid {}x{}:".format(num,num))
			for i in torchbearer.bestList:
				print(i)

			print("Scores:")
			for i in torchbearer.scoredList:
				print("{} dark squares: {}".format(i,torchbearer.scoredList[i]))

			rando = random.randint(0, len(torchbearer.bestList) - 1)
			print("Showing possible solution: {}".format(torchbearer.bestList[rando]))
			for i in torchbearer.bestList[rando]:
				torchbearer.teleport(agent_host,i[0],i[1])
				torchbearer.placeTorch(agent_host)

			agent_host.sendCommand("quit")
			time.sleep(0.1)

		'''
	# Iterate twice: once to check for the lowest length amongst solutions, the other to add them.
	# print(torchbearer.worst)
	if(len(torchbearer.worst) > 0):
		bestLength = len(torchbearer.worst)
	else:
		bestLength = 1
	# print(bestLength)
	for i in torchbearer.triedList:
		if(len(i) < bestLength):
			bestLength = len(i)
	# print(bestLength)
	for i in torchbearer.triedList:
		# print(i)
		# print(len(i))
		if(len(i) == bestLength):
			torchbearer.bestList.append(i)

	print("Best locations: ")
	for i in torchbearer.bestList:
		print(i)
		'''
