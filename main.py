from math import trunc
import numpy as np
import pygame
import networkx as nx
from UIpygame import PyUI as pyui
import matplotlib.pyplot as plt
import random
from Fraction import Fraction

Connections = []
SingleAgentConnections = []
## importing modules

TransMatrix = np.zeros((4,4),
                       dtype=np.int16).tolist()
TransMatrix[0][0] = Fraction(1, 2)
TransMatrix[1][0] = Fraction(1, 2)
TransMatrix[2][1] = Fraction(1)
TransMatrix[1][2] = Fraction(1,3)
TransMatrix[3][2] = Fraction(1,2)
TransMatrix[0][2] = Fraction(1,6)
TransMatrix[1][3]= Fraction(1,5)
TransMatrix[0][3]= Fraction(4,5)
if len(TransMatrix) != len(TransMatrix[0]):
    raise ValueError

NPTransMatrix = np.array(TransMatrix, dtype=np.float32)
statevector = [np.ones(len(TransMatrix))/len(TransMatrix)]
MatrixIters = [0]
pygame.init()
connectionCords = []
NumNodes = len(TransMatrix)
Nodes = [x for x in range(NumNodes)]
visitedTally = [0 for x in range(NumNodes)]

Agents = [[random.choice(Nodes),"", A := 25*np.array([np.cos(theta := 360*random.random()), np.sin(theta)]), np.array([0,0])] for x in range(100)]
SingleAgent = [[random.choice(Nodes)], 25*np.array([np.cos(theta := 360*random.random()), np.sin(theta)]), np.array([0,0])]
SingleAgentAnimating = [False]
AgentsAnimating = [False]
AgentAnimationFrames = [30]
SingleAgentAnimationFrames = [30]
CurrentFrame = [0]
CurrentSingleFrame = [0]

def posGen():
    num = 10
    while True:
        yield num
        num += 60


## setting up pygame and PyUI
screenw = 1200
screenh = 900
# creates screen objects of size screenw and screenh
# resizable tag allows the screen to be scaled, remove it to lock screen size
screen = pygame.display.set_mode((screenw, screenh),pygame.RESIZABLE)
# ui object is how most of the PyUI module is operated through
ui = pyui.UI()
# when done is set to True the gameloop ends
done = False
# clock keeps fps consistant at 60
clock = pygame.time.Clock()

ui.styleload_lightblue()
ui.styleset(textsize=50)
ui.styleset(textcenter=True)
ui.windowmenucol = pyui.shiftcolor(pyui.Style.objectdefaults[pyui.WINDOW]['col'],-35)


def changeMode():
    match OptionSelector.active:
        case "Matrix":
            ui.IDs["Msidebar"].open(toggleopen=False)
            ui.IDs["MBack"].open(animation="moveup", toggleopen=False, animationlength=60)
        case "T Avg":
            ui.IDs["Tsidebar"].open(toggleopen=False)
            ui.IDs["TBack"].open(animation="moveup", toggleopen=False,animationlength=60)
        case "Agents":
            ui.IDs["Asidebar"].open(toggleopen=False)
            ui.IDs["ABack"].open(animation="moveup", toggleopen=False,animationlength=60)

def applyMatrix(num = 1):
    for x in range(num):
        statevector[0] = NPTransMatrix @ statevector[0]
    MatrixIters[0] += num
    for rownum, entry in enumerate(statevector[0]):
        ui.IDs["StateVecEntry" + str(rownum)].settext(round(entry,5))
    ui.IDs["AppledNumber"].settext(MatrixIters[0])

def applyMatrixTen():
    applyMatrix(10)
def applyMatrixN():
    Iters = ui.IDs["NumIters"].text
    if (Value := int(Iters)) == float(Iters):
        applyMatrix(Value)

def setMatrixN():
    Iters = ui.IDs["SetNumIters"].text
    if (Value := int(Iters)) == float(Iters):
        statevector[0] = np.ones(len(TransMatrix))/len(TransMatrix)
        MatrixIters[0] = 0
        applyMatrix(Value)


def makeSideBar():


    #Make Matrix Menu Items
    ypos = posGen()
    MatrixItems = []
    MatrixItems.append(ui.makebutton(10,next(ypos),"Apply Matrix", command=applyMatrix,textsize=30, width=160,height=50))
    MatrixItems.append(ui.makebutton(10,next(ypos),"Apply Matrix 10x", command=applyMatrixTen, textsize=30, width=160,height=50))
    MatrixItems.append(ui.maketext(10,next(ypos),"Run Iters:", width=160,height=50,backingcol = ui.windowmenucol))
    MatrixItems.append(ui.maketextbox(10,next(ypos),15, ID = "NumIters", width=160,height=50,numsonly=True))
    MatrixItems.append(ui.makebutton(10,next(ypos),"Apply",command= applyMatrixN, textsize=40, width=160,height=50))
    MatrixItems.append(ui.maketext(10,next(ypos),"Set Iters:", width=160,height=50,backingcol=ui.windowmenucol))
    MatrixItems.append(ui.maketextbox(10,next(ypos),0,ID = "SetNumIters", width=160,height=50,numsonly=True))
    MatrixItems.append(ui.makebutton(10,next(ypos),"Apply",command=setMatrixN, textsize=40, width=160,height=50))
    ypos = posGen()
    TAvgItems = []
    TAvgItems.append(ui.makebutton(10,next(ypos)," Run 1 Step", command=doStep,textsize=40, width=160,height=50))
    TAvgItems.append(ui.maketext(10,next(ypos),"Frames:", width=160,height=50,backingcol=ui.windowmenucol))
    TAvgItems.append(ui.maketextbox(10,next(ypos),text="10", ID = "SingleAgentAnimTime",width=160,height=50,allowedcharacters=''.join([str(x) for x in range(10)])))
    TAvgItems.append(ui.makebutton(10,next(ypos),"Play/Pause",ID="PlaySingleAgentAgain",toggle=False,toggleable=True, command=doStepForToggle, textsize=40, width=160,height=50))
    TAvgItems.append(
        ui.maketext(10, next(ypos), "Skip Fowards:", textsize=30, width=160, height=50, backingcol=ui.windowmenucol))
    TAvgItems.append(ui.maketextbox(10, next(ypos), "0", ID="SingleAgentSkipIters", width=160, height=50, numsonly=True))
    TAvgItems.append(ui.makebutton(10,next(ypos),"Apply", command=skipAgentForwards, textsize=40, width=160,height=50))
    TAvgItems.append(ui.maketext(10,next(ypos),"Start Node:",textcenter=True, textsize=40,width=160,height=50,backingcol=ui.windowmenucol))
    TAvgItems.append(ui.maketextbox(10,next(ypos), width=160,ID="StartNode",height=50,allowedcharacters=[chr(65+x) for x in range(NumNodes)]))
    TAvgItems.append(ui.makebutton(10,next(ypos),"Restart",command= restartAgents, textsize=40, width=160,height=50))

    ypos = posGen()
    AgentAvg = []
    AgentAvg.append(ui.makebutton(10,next(ypos)," Run 1 Step",command=doStep, textsize=40, width=160,height=50))
    AgentAvg.append(ui.maketext(10,next(ypos),"Frames:", width=160,height=50,backingcol=ui.windowmenucol))
    AgentAvg.append(ui.maketextbox(10,next(ypos), ID = "AgentAnimTime",text=30,width=160,height=50,allowedcharacters=''.join([str(x) for x in range(10)])))
    AgentAvg.append(ui.makebutton(10,next(ypos),"Play/Pause",ID="PlayAgentsAgain",toggle=False, command=doStepForToggle,textsize=40,toggleable=True, width=160,height=50))
    AgentAvg.append(ui.maketext(10,next(ypos),"Skip Fowards:",textsize=30, width=160,height=50,backingcol=ui.windowmenucol))
    AgentAvg.append(ui.maketextbox(10,next(ypos),"0",ID="AgentSkipIters", width=160,height=50,numsonly=True))
    AgentAvg.append(ui.makebutton(10,next(ypos),"Apply", command=skipAgentForwards, textsize=40, width=160,height=50))
    AgentAvg.append(ui.maketext(10,next(ypos),"Num Agents:", width=160,textsize=40,height=50,backingcol=ui.windowmenucol))
    AgentAvg.append(ui.maketextbox(10,next(ypos),text="100",ID = "AgentNum", width=160,height=50,allowedcharacters=''.join([str(x) for x in range(10)])))
    AgentAvg.append(ui.makebutton(10,next(ypos),"Restart",command= restartAgents, textsize=40, width=160,height=50))
    #Make Menus in top right
    for item in (MatrixItems+AgentAvg+TAvgItems):
        item.scaleby = "vertical"
    MatrixMenu = ui.makewindow(-190, 70, 180,500, ID= "Msidebar", anchor=('w', '0'),bounditems=MatrixItems, animationtype='moveright',autoshutwindows=["Tsidebar", "Asidebar"], scaleby="vertical")
    TimeMenu = ui.makewindow(-190, 70, 180,640, ID= "Tsidebar", anchor=('w', '0'),bounditems=TAvgItems,animationtype= 'moveright',autoshutwindows=["Msidebar", "Asidebar"],scaleby="vertical")
    AgentMenu = ui.makewindow(-190, 70, 180,640, ID= "Asidebar", anchor=('w', '0'),bounditems= AgentAvg,animationtype= 'moveright',autoshutwindows=["Tsidebar", "Msidebar"],scaleby="vertical")
    MatrixMenu.open(animation='left')



def makeMatrixBacking():
    ypos = posGen()
    MatrixBackingItems = []
    MatrixBackingItems.append(ui.maketext(150,220,"P =", textsize=100,centery=True, textcenter=False))
    MatrixBackingItems.append(ui.makerect(285,10,5,420, col =pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(285, 10, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(285, 425, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(700, 10, 5, 420, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(685, 10, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(685, 425, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    spacing = 420/(len(TransMatrix)+1)
    for rownum, row in enumerate(TransMatrix):
        for columnnum, entry in enumerate(row):
            entryText = ui.maketext(285+(columnnum+1)*spacing,10 + (rownum+1)* spacing,entry, center=True,textsize=int(250/(len(TransMatrix))))
            MatrixBackingItems.append(entryText)
    MatrixBackingItems.append(ui.maketext(150,680,"P", textsize=100,centery=True, textcenter=False , bounditems=
                                          [ui.maketext(0, 0, "0",ID="AppledNumber", textsize=50, centery=False, textcenter=False, scaleby="vertical", objanchor=('0','h/2'), anchor=('w','0'))]
                                          ))
    MatrixBackingItems.append(ui.maketext(240, 680, "π", textsize=100, centery=True, textcenter=False, bounditems=
    [ui.maketext(0, 0, "0", textsize=50, centery=False, textcenter=False, scaleby="vertical", objanchor=('w/2', 'h/2'),
                 anchor=('w', 'h'))]
                                          ))
    MatrixBackingItems.append(ui.maketext(400, 680, "=", textsize=100, centery=True, textcenter=False))
    MatrixBackingItems.append(ui.makerect(530,450,5,420, col =pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(530, 450, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(530, 865, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(720, 450, 5, 420, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(705, 450, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))
    MatrixBackingItems.append(ui.makerect(705, 865, 20, 5, col=pyui.Style.defaults["textcol"], roundedcorners=False))

    for rownum, entry in enumerate(statevector[0]):
        entryText = ui.maketext(625,450 + (rownum+1)* spacing,round(entry,5),ID="StateVecEntry" + str(rownum), center=True,textsize=int(250/(len(TransMatrix))))
        MatrixBackingItems.append(entryText)

    for item in MatrixBackingItems:
        item.scaleby = "vertical"
    #Show Graph or Matrix
    MatrixBacking = ui.makewindow(0, 10, 1000-10,880, ID= "MBack",bounditems=MatrixBackingItems, animationtype='movedown',autoshutwindows=["TBack", "ABack"],backingdraw=False,scaleby="vertical", objanchor=('w/2','0'), anchor=('(w-200)/2', '0'))
    MatrixBacking.open(animation='moveup')

def makeAgentBacking():
    AgentBackingItems = []
    G = nx.DiGraph()
    G.add_nodes_from([x for x in range(NPTransMatrix.shape[0])])
    for rownum, row in enumerate(NPTransMatrix):
        for colnum, val in enumerate(row):
            if val != 0:
                G.add_edge(colnum, rownum)
                Connections.append((colnum, rownum, val))
                AgentBackingItems.append(ui.maketext(0,0,str(round(val,4)),20, ID = "AgentConnection" + str(len(Connections)-1)))
    ypos = posGen()
    for pos in list(nx.spring_layout(G).items()):
        AgentBackingItems.append(ui.makebutton(pos[1][0]*450,pos[1][0]*400,chr(pos[0]+65),width=50,height=50, hovercol=0,roundedcorners=25,clickdownsize=0,dragable=True,ID= "AgentNode"+ str(pos[0]), runcommandat=1, objanchor=('w/2', 'h/2') , anchor=('w/2', 'h/2')))
        AgentBackingItems.append(ui.maketext(0, ycord := next(ypos)/2,chr(pos[0]+65)))
        AgentBackingItems.append(ui.makerect(40,ycord+10,ID= "AgentBar"+ str(pos[0]),width = 300*len(list(filter(lambda x: x[0] == pos[0], Agents)))/len(Agents),height= 20))
    for item in AgentBackingItems:
        item.scaleby = "vertical"
    AgentBacking = ui.makewindow(10, 10, 1000-10,880, ID= "ABack",bounditems=AgentBackingItems,backingdraw=False,animationtype= 'movedown',autoshutwindows=["TBack", "MBack"],scaleby="vertical", objanchor=('w/2','0'), anchor=('(w-200)/2', '0'))


def makeTimeBacking():
    TimeBackingItems = []
    G = nx.DiGraph()
    G.add_nodes_from([x for x in range(NPTransMatrix.shape[0])])
    for rownum, row in enumerate(NPTransMatrix):
        for colnum, val in enumerate(row):
            if val != 0:
                G.add_edge(colnum, rownum)
                SingleAgentConnections.append((colnum, rownum, val))
                TimeBackingItems.append(
                    ui.maketext(0, 0, str(round(val, 4)), 20, ID="SingleAgentConnection" + str(len(SingleAgentConnections) - 1)))
    ypos = posGen()
    for pos in list(nx.spring_layout(G).items()):
        TimeBackingItems.append(ui.makebutton(pos[1][0]*450,pos[1][0]*400,chr(pos[0]+65),width=50,height=50, hovercol=0,roundedcorners=25,clickdownsize=0,dragable=True,ID= "SingleAgentNode"+ str(pos[0]), runcommandat=1, objanchor=('w/2', 'h/2') , anchor=('w/2', 'h/2')))
        TimeBackingItems.append(ui.maketext(0, ycord := next(ypos)/2,chr(pos[0]+65)))
        TimeBackingItems.append(ui.makerect(40,ycord+10,ID= "SingleAgentBar"+ str(pos[0]),width = 0,height= 20))
    for item in TimeBackingItems:
        item.scaleby = "vertical"
    TimeBacking = ui.makewindow(10, 10, 1000-10,880, ID= "TBack",bounditems = TimeBackingItems, scaleby="vertical",backingdraw=False,animationtype= 'movedown',autoshutwindows=["MBack", "ABack"], objanchor=('w/2','0'), anchor=('(w-200)/2', '0'))


def connectNodes():
    if OptionSelector.active == "Agents":
        yoffset = 0
        try:
            yoffset = ui.IDs["ABack"].yoff
        except:
            pass
        for connectionNum,connection in enumerate(Connections):
            Start = ui.IDs["AgentNode"+ str(connection[0])]
            scalex = Start.dirscale[0]
            scaley = Start.dirscale[1]
            scaleMat = np.array([[scalex,0],[0,scaley]])
            StartCords = np.array([(scalex * Start.x+scaley*25), (Start.y+25+yoffset)*scaley])
            End = ui.IDs["AgentNode"+ str(connection[1])]
            EndCords = np.array([(End.x*scalex+25*scaley), (End.y+25+yoffset)*scaley])
            StartToEnd = (EndCords - StartCords)
            if (normalfactor := np.linalg.norm(StartToEnd,2)) == 0:
                 StartCords = (StartCords+np.array([0,-scaley])*25)
                 EndCords  = (EndCords+np.array([-scaley,0])*25)
                 setOfPoints = [StartCords, (StartCords+ (np.array([0,-scaley])*70)), (EndCords+ (np.array([-scaley,0])*70)), EndCords]
            else:
                if NPTransMatrix[connection[0]][connection[1]] == 0:
                    StartCords = (StartCords + scaley* StartToEnd * 25 / normalfactor)
                    EndCords = (EndCords - scaley * StartToEnd * 25 / normalfactor)
                    setOfPoints = [StartCords, EndCords]

                else:
                    CurvePoint = 1/5* np.array([[0,scaley],[-scaley,0]]) @ StartToEnd/2 + (StartCords + StartToEnd/2)
                    StartCords = (StartCords+ scaley* StartToEnd*25/normalfactor)
                    EndCords = (EndCords - scaley*StartToEnd * 25/normalfactor)
                    setOfPoints = [StartCords,CurvePoint, EndCords]

            #pygame.draw.line(screen, (0,0,0),StartCords,EndCords,2)

            #print(pyui.draw.bezierdrawer((StartCords,EndCords),2,commandpoints=False))
            pointsOnLine = pyui.draw.bezierdrawer(setOfPoints,2,commandpoints=False,rounded=False)
            pygame.draw.lines(screen, color= (0,0,0), closed = False, points=  pointsOnLine)
            ui.IDs["AgentConnection" + str(connectionNum)].x = pointsOnLine[20][0]/scalex +5
            ui.IDs["AgentConnection" + str(connectionNum)].y = pointsOnLine[20][1]/scaley +5
            # Make Arrows
            ArrowLen =15
            ArrowWidth = 4
            ArrowBase, ArrowHead = [np.array(x,dtype=np.float64) for x in (pointsOnLine[-2:])]
            ArrowSpine = ArrowHead - ArrowBase
            ArrowSpine /= np.linalg.norm(ArrowSpine,2)
            leftArrow = ArrowHead - ArrowSpine * ArrowLen+ ArrowWidth *np.array([[0,1],[-1,0]]) @ ArrowSpine
            rightArrow = ArrowHead - ArrowSpine* ArrowLen + ArrowWidth * np.array([[0, -1], [1, 0]]) @ ArrowSpine
            pyui.draw.polygon(screen, (0,0,0), [ArrowHead,leftArrow,rightArrow])
    #Im very aware i have copy pasted its 2am idc anymore

    elif OptionSelector.active == "T Avg":
        yoffset = 0
        try:
            yoffset = ui.IDs["TBack"].yoff
        except:
            pass
        for connectionNum,connection in enumerate(SingleAgentConnections):
            Start = ui.IDs["SingleAgentNode" + str(connection[0])]
            scalex = Start.dirscale[0]
            scaley = Start.dirscale[1]
            scaleMat = np.array([[scalex, 0], [0, scaley]])
            StartCords = np.array([(Start.x *scalex+ 25*scaley), scaley*(Start.y + 25 + yoffset)])
            End = ui.IDs["SingleAgentNode" + str(connection[1])]
            EndCords = np.array([(End.x*scalex + 25*scaley), scaley*(End.y + 25 + yoffset)])
            StartToEnd = (EndCords - StartCords)
            if (normalfactor := np.linalg.norm(StartToEnd,2)) == 0:
                 StartCords = (StartCords+np.array([0,-scaley])*25)
                 EndCords  = (EndCords+np.array([-scaley,0])*25)
                 setOfPoints = [StartCords, (StartCords+ (np.array([0,-scaley])*70)), (EndCords+ (np.array([-scalex,0])*70)), EndCords]

            else:
                if NPTransMatrix[connection[0]][connection[1]] == 0:
                    StartCords = (StartCords + scaley* StartToEnd * 25 / normalfactor)
                    EndCords = (EndCords - scaley* StartToEnd * 25 / normalfactor)
                    setOfPoints = [StartCords, EndCords]
                else:
                    CurvePoint = 1/5*np.array([[0,scaley],[-scaley,0]]) @ StartToEnd/2 + (StartCords + StartToEnd/2)
                    StartCords = (StartCords+scaley* StartToEnd*25/normalfactor)
                    EndCords = (EndCords - scaley* StartToEnd * 25/normalfactor)
                    setOfPoints = [StartCords,CurvePoint, EndCords]

            # print(pyui.draw.bezierdrawer((StartCords,EndCords),2,commandpoints=False))
            pointsOnLine = pyui.draw.bezierdrawer(setOfPoints, 2, commandpoints=False, rounded=False)
            pygame.draw.lines(screen, color=(0, 0, 0), closed=False, points=pointsOnLine)

            ui.IDs["SingleAgentConnection" + str(connectionNum)].x = pointsOnLine[20][0]/scalex + 5
            ui.IDs["SingleAgentConnection" + str(connectionNum)].y = pointsOnLine[20][1]/scaley + 5
            # Make Arrows
            ArrowLen = 15
            ArrowWidth = 4
            ArrowSpine = np.array([], dtype=np.float64)
            ArrowBase, ArrowHead = [np.array(x, dtype=np.float64) for x in (pointsOnLine[-2:])]
            ArrowSpine = ArrowHead - ArrowBase
            ArrowSpine /= np.linalg.norm(ArrowSpine, 2)
            leftArrow = ArrowHead - ArrowSpine * ArrowLen + ArrowWidth * np.array([[0, 1], [-1, 0]]) @ ArrowSpine
            rightArrow = ArrowHead - ArrowSpine * ArrowLen + ArrowWidth * np.array([[0, -1], [1, 0]]) @ ArrowSpine
            pyui.draw.polygon(screen, (0, 0, 0), [ArrowHead, leftArrow, rightArrow])

def drawAgents():
    if OptionSelector.active == "Agents":
        yoffset = 0
        try:
            yoffset = ui.IDs["ABack"].yoff
        except:
            pass
        scalex = ui.IDs["AgentNode0"].dirscale[0]
        scaley = ui.IDs["AgentNode0"].dirscale[1]
        scaleMat = np.array([[scalex, 0], [0, scaley]])
        if not(AgentsAnimating[0]):
            for Agent in Agents:
                Start = ui.IDs["AgentNode"+ str(Agent[0])]
                StartCords = (np.array([(Start.x *scalex + 25*scaley), scaley*(Start.y + 25+ yoffset)])+ scaley * Agent[2])
                pyui.draw.circle(screen, (255,20,100),StartCords, 3)
            return
        for Agent in Agents:
            Start = ui.IDs["AgentNode" + str(Agent[1])]
            End = ui.IDs["AgentNode" + str(Agent[0])]
            StartCords =  (np.array([(Start.x*scalex + 25*scaley),scaley* (Start.y + 25 + yoffset)]) + scaley*Agent[3])
            EndCords =  (np.array([(scalex*(End.x) + scaley*(25)), scaley*(End.y + 25 + yoffset)]) + scaley* Agent[2])
            pyui.draw.circle(screen, (255, 20, 100), StartCords - (CurrentFrame[0]/AgentAnimationFrames[0]) * (StartCords-EndCords), 3*scaley)
        CurrentFrame[0] += 1
        if CurrentFrame[0] == AgentAnimationFrames[0]:
            CurrentFrame[0] = 0
            AgentsAnimating[0] = False
            if ui.IDs["PlayAgentsAgain"].toggle:
                doStep()
    elif OptionSelector.active == "T Avg":
        yoffset = 0
        try:
            yoffset = ui.IDs["TBack"].yoff
        except:
            pass
        scalex = ui.IDs["SingleAgentNode0"].dirscale[0]
        scaley = ui.IDs["SingleAgentNode0"].dirscale[1]
        scaleMat = np.array([[scalex, 0], [0, scaley]])
        if not (SingleAgentAnimating[0]):
            Start = ui.IDs["SingleAgentNode" + str(SingleAgent[0][-1])]
            StartCords =  (np.array([(scalex* Start.x + scaley*25), scaley*(Start.y + 25 + yoffset)]) + scaley*SingleAgent[1])
            pyui.draw.circle(screen, (255, 20, 100), StartCords, 3)
            return
        Start = ui.IDs["SingleAgentNode" + str(SingleAgent[0][-2])]
        End = ui.IDs["SingleAgentNode" + str(SingleAgent[0][-1])]
        StartCords = (np.array([(scalex*Start.x + scaley*25), scaley*(Start.y + 25 + yoffset)]) + scaley*SingleAgent[2])
        EndCords =  (np.array([(scalex*End.x + scaley*25), scaley*(End.y + 25 + yoffset)]) +scaley* SingleAgent[1])
        pyui.draw.circle(screen, (255, 20, 100), StartCords - (CurrentSingleFrame[0] / SingleAgentAnimationFrames[0]) * (StartCords - EndCords), 3)
        CurrentSingleFrame[0] += 1
        if CurrentSingleFrame[0] == SingleAgentAnimationFrames[0]:
            CurrentSingleFrame[0] = 0
            SingleAgentAnimating[0] = False
            if ui.IDs["PlaySingleAgentAgain"].toggle:
                doStep()

def doStepForToggle():
    if OptionSelector.active == "Agents":
        if ui.IDs["PlayAgentsAgain"].toggle:
            doStep()
    elif OptionSelector.active == "T Avg":
        if ui.IDs["PlaySingleAgentAgain"].toggle:
            doStep()

def doStep(animate=True):
    if OptionSelector.active == "Agents":
        if AgentsAnimating[0]:
            return
        if animate:
            AgentsAnimating[0]= True
        try:
            if (frames := int(ui.IDs["AgentAnimTime"].text)) !=0:
                AgentAnimationFrames[0] = frames
        except: pass
        for Agent in Agents:
            currentPos = Agent[0]
            choice = random.random()
            for destin in range(NumNodes):
                choice -= NPTransMatrix[destin][currentPos]
                if choice <= 0:
                    break

            Agent[0],Agent[1] = destin, Agent[0]
            Agent[2], Agent[3] = 25*np.array([np.cos(theta := 360*random.random()), np.sin(theta)]), Agent[2]
        for updateBar in range(NumNodes):
            ui.IDs["AgentBar"+ str(updateBar)].width = 300*len(list(filter(lambda x: x[0] == updateBar, Agents)))/len(Agents)

    elif OptionSelector.active == "T Avg":
        if SingleAgentAnimating[0]:
            return
        if animate:
            SingleAgentAnimating[0] = True
        try:
            if (frames := int(ui.IDs["SingleAgentAnimTime"].text)) != 0:
                SingleAgentAnimationFrames[0] = frames
        except: pass
        currentPos = SingleAgent[0][-1]
        choice = random.random()
        for destin in range(NumNodes):
            choice -= NPTransMatrix[destin][currentPos]
            if choice <= 0:
                break
        SingleAgent[0].append(destin)
        SingleAgent[1], SingleAgent[2] = 25 * np.array([np.cos(theta := 360 * random.random()), np.sin(theta)]), SingleAgent[1]
        visitedTally[destin] += 1
        for updateBar in range(NumNodes):
            ui.IDs["SingleAgentBar" + str(updateBar)].width = 300 * visitedTally[updateBar]/sum(visitedTally)
def skipAgentForwards():
    if OptionSelector.active == "Agents":
        try:
            for x in range(int(ui.IDs["AgentSkipIters"].text)):
                doStep(False)
        except:
            pass
    elif OptionSelector.active == "T Avg":
        try:
            for x in range(int(ui.IDs["SingleAgentSkipIters"].text)):
                doStep(False)
        except:
            pass

def restartAgents():
    if OptionSelector.active == "Agents":
        AgentsAnimating[0] = False
        CurrentFrame[0] = 0
        try:
            newAgentNum = int(ui.IDs["AgentNum"].text)

        except:
            return
        if newAgentNum >= 1:
            Agents.clear()
            for x in range(newAgentNum):
                Agents.append([random.choice(Nodes),"", A := 25*np.array([np.cos(theta := 360*random.random()), np.sin(theta)]), np.array([0,0])])
        if ui.IDs["PlayAgentsAgain"].toggle:
            doStep()
        for updateBar in range(NumNodes):
            ui.IDs["AgentBar"+ str(updateBar)].width = 300*len(list(filter(lambda x: x[0] == updateBar, Agents)))/len(Agents)
    elif OptionSelector.active == "T Avg":
        SingleAgentAnimating[0] = False
        CurrentSingleFrame[0] = 0
        newPos = ui.IDs["StartNode"].text
        if not(ord(newPos)-65 in Nodes):
            newPos = random.choice(Nodes)
        else:
            newPos = ord(newPos)-65
        SingleAgent[0] = [newPos]
        if ui.IDs["PlaySingleAgentAgain"].toggle:
            doStep()
        for updateBar in range(NumNodes):
            ui.IDs["SingleAgentBar" + str(updateBar)].width = 0


OptionSelector = ui.makedropdown(-190, 10, ["Matrix", "T Avg", "Agents"], width=180, height=50, command=changeMode,
                                     anchor=('w', '0'), dropsdown=False,scaleby="vertical")
makeSideBar()
makeAgentBacking()
makeTimeBacking()
makeMatrixBacking()



# main game loop
while not done:
    # grabs event data like button inputs and mouse position and ruturns pygame event data
    # can be treated the same as pygame.event.get() function
    for event in ui.loadtickdata():
        if event.type == pygame.QUIT:
            done = True
    # fills screen with the wallpaper col, defaults to white
    screen.fill(pyui.Style.wallpapercol)



    # draws and processes all gui objects

    ui.rendergui(screen)
    connectNodes()
    drawAgents()
    # displays all changes to the monitor screen
    pygame.display.flip()
    # maintains 60 fps
    clock.tick(60)
# shuts pygame window
pygame.quit()