from tkinter import filedialog
import pickle


def saveSetupClean(self, FILE="", returnSer = False):

    if not returnSer:
        if not FILE:
            FILE = filedialog.asksaveasfile(
                mode='wb',
                defaultextension=".pkl",
                confirmoverwrite=True,
                filetypes=[("Pickle files", "*.pkl")],  # Only allow .pkl files
                title="Save node preset",
                initialdir=f"{self.MAINPATH}/presets"
            )

        else:
            FILE = open(FILE, mode="wb")
        if not FILE: return

    serialized = []
    id_counter = 0
    io_id_counter = 0

    for node in self.NODES:
        node.id = id_counter
        id_counter += 1
        for io in node.inputs + node.outputs:
            io.id = io_id_counter
            io_id_counter += 1

    for node in self.NODES:
        node_data = {
            "id": node.id,
            "fname": node.ARGS[2],
            "pos": node.pos,
            "utility": [x.value for x in node.utility],
            "io": []
        }

        for io in node.inputs + node.outputs:
            if io.selfInput:
                continue
            io_data = {
                "id": io.id,
                "is_output": io.isOutput,
                "connection": None
            }
            if io.IN_CONNECTION:
                io_data["connection"] = {
                    "target_node_id": io.IN_CONNECTION.parent.id,
                    "target_io_id": io.IN_CONNECTION.id
                }
            node_data["io"].append(io_data)
        serialized.append(node_data)

    if returnSer:
        return serialized

    pickle.dump(serialized, FILE)
    FILE.close()
    self.notify(f"Setup {FILE.name} saved!")


def loadSetupClean(self, loadTo, FILE="", serialized=None):

    if not serialized:

        if not FILE:
            FILE = filedialog.askopenfile(
                mode='rb',
                defaultextension=".pkl",
                filetypes=[("Preset files", "*.pkl")],  # Only allow .pkl files
                title="Load node preset",
                initialdir=f"{self.MAINPATH}/presets"
            )

        else:
            FILE = open(FILE, mode="rb")
        if not FILE: return

    loadTo.clear()
    try:
        if serialized:
            data = serialized
        else:
            data = pickle.load(FILE)

        # Pass 1: Create all nodes, assign IDs
        node_lookup = {}
        io_lookup = {}

        for nd in data:
            node = self.NODEBLUEPRINTS[nd["fname"]].copy()
            node.realPos = nd["pos"]
            for i, v in enumerate(nd["utility"]):
                node.utility[i].value = v
            node.id = nd["id"]

            i = 0
            for io in node.inputs + node.outputs:
                if io.selfInput:
                    continue
                io.id = nd["io"][i]["id"]
                io_lookup[io.id] = io
                i += 1

            node_lookup[node.id] = node
            node.addTo(loadTo)

        # Pass 2: Connect inputs
        for nd in data:
            node = node_lookup[nd["id"]]
            for io_def in nd["io"]:
                if io_def["is_output"] or not io_def["connection"]:
                    continue
                this_io = io_lookup[io_def["id"]]
                target_io = io_lookup[io_def["connection"]["target_io_id"]]
                this_io.connect(target_io)
        if not serialized:
            self.notify(f"Setup {FILE.name} loaded!")
        self.EXPORT = False

    except Exception as e:
        self.notify(f"Error loading setup: {e}")

