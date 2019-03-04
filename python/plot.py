import matplotlib.pyplot as plt
import csv
import sys

PLOT_COUNT = 4

def plot_accuracy(filename):
    X = []
    multiY = [[] for i in range(PLOT_COUNT)]

    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            X.append(float(row[0]))
            multiY[0].append(float(row[1]))
            multiY[1].append(float(row[3]))
            multiY[2].append(float(row[5]))
            multiY[3].append(float(row[7]))

    plt.figure()
    plt.grid(True)

    colors=('r', 'c', 'm', 'k')
    i=0
    for Y in reversed(multiY):
        plt.plot(X, Y,colors[i])
        i+=1


    plt.title("Accuracy, function of the Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.legend(["MIMIC", "GA", "SA", "RHC"])
    plt.show()

def plot_compute_time(filename):
    X = []
    multiY = [[] for i in range(PLOT_COUNT)]

    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            X.append(float(row[0]))
            multiY[0].append(float(row[2]))
            multiY[1].append(float(row[4]))
            multiY[2].append(float(row[6]))
            multiY[3].append(float(row[8]))

    plt.figure()
    plt.grid(True)
    colors=('r', 'c', 'm', 'k')
    i=0
    for Y in reversed(multiY):
        plt.plot(X, Y,colors[i])
        i+=1
    plt.title("Computational Time, function of the Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Computational Time")
    plt.legend(["MIMIC", "GA", "SA", "RHC"])
    plt.show()

def plot_accuracy_compute_time(filename):
    ALGORITHMS = ["MIMIC", "GA", "SA", "RHC"]
    COLORS =('r', 'c', 'm', 'k')

    multiPair = [[] for i in range(PLOT_COUNT)]

    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            multiPair[0].append((float(row[2]), float(row[1])))
            multiPair[1].append((float(row[4]), float(row[3])))
            multiPair[2].append((float(row[6]), float(row[5])))
            multiPair[3].append((float(row[8]), float(row[7])))

    plt.figure()
    plt.grid(True)
    for i, pairList in enumerate(reversed(multiPair)):
        plt.subplot(2, 2, i + 1)
        pairList.sort(key=lambda x: x[0])
        plt.plot(
            [pairList[i][0] for i in range(len(pairList))],
            [pairList[i][1] for i in range(len(pairList))],
            COLORS[i]
        )
        plt.title("Accuracy, function of the Computational Time")
        plt.xlabel("Computational Time")
        plt.ylabel("Accuracy")
        plt.legend([ALGORITHMS[i]])

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(
            "Error ! Usage: python3 plot.py plot_type filename\n" + 
            "plot_type: [\"accuracy\", \"compute_time\", \"accuracy_compute_time\"]"
        )
    else:
        filename = str(sys.argv[2])
        type = str(sys.argv[1])
        
        if type == "accuracy":
            plot_accuracy(filename)
        elif type == "compute_time":
            plot_compute_time(filename)
        elif type == "accuracy_compute_time":
            plot_accuracy_compute_time(filename)
        else:
            print(
                "Error ! Usage: python3 plot.py plot_type filename\n" + 
                "plot_type: [\"accuracy\", \"compute_time\", \"accuracy_compute_time\"]"
            )