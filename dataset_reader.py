import random

GAMMAS_SIZE = 12332
HADRONS_SIZE = 6688


class DataReader():
    def __init__(self, file_path):
        self.file = open(file_path, 'r')

    def read(self):
        """
        1. fLength: continuous # major axis of ellipse [mm]
        2. fWidth: continuous # minor axis of ellipse [mm]
        3. fSize: continuous # 10-log of sum of content of all pixels [in #phot]
        4. fConc: continuous # ratio of sum of two highest pixels over fSize [ratio]
        5. fConc1: continuous # ratio of highest pixel over fSize [ratio]
        6. fAsym: continuous # distance from highest pixel to center, projected onto major axis [mm]
        7. fM3Long: continuous # 3rd root of third moment along major axis [mm]
        8. fM3Trans: continuous # 3rd root of third moment along minor axis [mm]
        9. fAlpha: continuous # angle of major axis with vector to origin [deg]
        10. fDist: continuous # distance from origin to center of ellipse [mm]
        11. class: g,h # gamma (signal), hadron (background)

        g = gamma (signal): 12332
        h = hadron (background): 6688
        """
        samples, labels = [], []
        gammas = []
        hadrons = []
        line = self.file.readline()[:-1]  # to remove \n
        while len(line) != 0:
            features = line.split(',')
            if features[-1] == 'g':
                gammas.append(([float(features[i]) for i in range(10)], features[10]))
            else:
                hadrons.append(([float(features[i]) for i in range(10)], features[10]))
            line = self.file.readline()[:-1]
        random.shuffle(gammas)
        for i in range(HADRONS_SIZE):
            samples.append(gammas[i][0])
            samples.append(hadrons[i][0])
            labels.append(gammas[i][1])
            labels.append(hadrons[i][1])
        self.file.close()
        return samples, labels
