import re
import numpy as np
class FOV:
    # Conversion factors to µm
    UNIT_CONVERSIONS = {
        "mm": 1000,  # 1 mm = 1000 µm
        "µm": 1,     # 1 µm = 1 µm
        "nm": 0.001  # 1 nm = 0.001 µm
    }

    def __init__(self, measurements, image):
        """
        Initialize the FieldOfView object by parsing the provided measurement strings.

        :param measurements: List of measurement strings (e.g., ['387.9µm', '290.9µm'])
        :param image: Image array to get resolution
        """
        self.values = []
        self.unit = "µm"  # Standard unit
        if isinstance(image, np.ndarray):
            self.resolution = image.shape[0:2]
        else:
            self.resolution = image

        self._parse_measurements(measurements)
        a, u = self.getAreaOfPixel()

    def getAreaOfPixel(self):
        x = self.values[0] / self.resolution[0]
        y = self.values[1] / self.resolution[1]
        return x * y, self.unit

    def getPixelToDistance(self):
        return self.values[0] / self.resolution[0]
    
    def getAreaOfImage(self):
        return self.values[0] * self.values[1]
    
    def getResolution(self, requestedUnit = "µm"):
        return [x/self.UNIT_CONVERSIONS[requestedUnit] for x in self.values]

    def _parse_measurements(self, measurements):
        """
        Parses the input measurement strings, converts to µm, and stores numeric values.

        :param measurements: List of measurement strings
        """
        for measurement in measurements:
            match = re.match(r"([\d.]+)([a-zA-Zµ]+)", measurement)
            if match:
                value, unit = match.groups()
                value = float(value)

                # Convert to µm if needed
                if unit in self.UNIT_CONVERSIONS:
                    if unit != "µm":
                        print("CONVERTING", unit)
                    value *= self.UNIT_CONVERSIONS[unit]  # Convert to µm
                else:
                    raise ValueError(f"Unknown unit: '{unit}'")

                self.values.append(value)  # Store converted value
                self.unit = "µm"

    def __repr__(self):
        return f"FieldOfView(values={self.values}, unit='{self.unit}'), {self.resolution[0] / self.values[0]}, {self.resolution[1] / self.values[1]} px/{self.unit}, Area of a pixel: {self.getAreaOfPixel()}"
