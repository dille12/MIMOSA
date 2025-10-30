


class inputNodeBP:
    def __init__(self, app, IO, connection = None):
        
        
        IO.id = app.idHelper
        self.id = app.idHelper

        print("IO CREATED FOR", IO.parent.name, self.id)

        app.idHelper += 1
        self.connection = None
        if connection:
            print("Searching connection.", self.id, connection.id)
            CONNECTED = False
            for x in app.nodeSkeletons:
                for y in x.IO:
                    if y.id == connection.id:
                        self.connection = y
                        CONNECTED = True
                        print(self.connection)
                        break

            print("CONNECTED:", CONNECTED)

class nodeBluePrint:
    def __init__(self, node, app):
        print("Saving", node.name)
        if node in app.nodesSaved:
            print("Returing")
            return
        
        
        app.nodesSaved.append(node)
        app.nodeSkeletons.append(self)
        
        self.pos = node.pos
        print(node.ARGS)
        self.FNAME = node.ARGS[2]
        print(self.FNAME)
        self.utilityVal = []
        for x in node.utility:
            self.utilityVal.append(x.value)


        self.IO = []

        for x in node.inputs + node.outputs:

            if x.selfInput:
                continue

            self.IO.append(inputNodeBP(app, x))

            for y in [x.IN_CONNECTION]:
                if not y:
                    continue
                
                y.parent.save()

        print("BEGIN CONNECTION")
        for x in node.inputs:
            if x.selfInput:
                continue

            #print(x.id, x.IN_CONNECTION.id)
            ownNode = self.matchIO(app, x)
            outNode = self.matchIO(app, x.IN_CONNECTION)
            if ownNode and outNode:
                ownNode.connection = outNode
                outNode.connection = ownNode

        

    def matchIO(self, app, InputNode):

        if not hasattr(InputNode, "id"):
            return

        for x in app.nodeSkeletons:
            for y in x.IO:

                if not hasattr(y, "id"):
                    continue

                if InputNode.id == y.id:
                    return y
                
        print("NO INPUT NODE FOUND FOR ID", InputNode.id)




    def build(self, app, addTo):

        nodeTemp = app.NODEBLUEPRINTS[self.FNAME].copy()
        nodeTemp.realPos = self.pos
        try:
            for i, x in enumerate(self.utilityVal):
                nodeTemp.utility[i].value = x
        except:
            pass

        i = 0

        

        for x in nodeTemp.inputs + nodeTemp.outputs:
            if x.selfInput:
                continue

            print("IO NODE:", x.typeC, x.parent.name, x.isOutput)
            
            x.id = self.IO[i].id
            print(x.id)
            if not self.IO[i].connection:
                continue
            print("CONNECTION:", self.IO[i].connection, "ID", self.IO[i].connection.id)

            mNode = self.matchNode(self.IO[i].connection.id, addTo)
            if mNode:
                x.connect(mNode)
                print("CONNECTED")
            else:
                print("Node doesn't exist yet.")

            i += 1


        addTo.append(nodeTemp)

    def matchNode(self, id, addTo):
        for x in addTo:
            for y in x.inputs + x.outputs:

                if not hasattr(y, "id"):
                    continue

                if y.id == id:
                    return y
        

    def __str__(self):
        for x in self.IO:
            print(x)

        return(f"NODE SKELETON: {self.FNAME}")

    
