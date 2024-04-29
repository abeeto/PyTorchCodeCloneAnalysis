import csv
import ast
currentDevice = {}

multipleDevices = {
    "Tesla K80" : [300, 4.113, '' , 13.71,  ''],
    "Tesla V100-PCIE-16GB" : [300, 14.13, 28.26, 4.71, 94.2],
    "T4" : [70, 8.141, 65.13, 116.3, 930],
}


def calculate(epochs):
    with open('emissions.csv', 'r') as file:
        reader = csv.reader(file)
        for x in range(2):
            head = next(reader)
            if(x != 0):
                time = ((float(head[3]))*float(epochs))/3600 #hours
                # print(time)
                emission = (float(head[4])*float(epochs))
                # print(emission)
                energyConsumed = (float(head[5])*float(epochs))
                # print(energyConsumed)
                currentDevice['TotalTime'] = time
                currentDevice['energyConsumed'] = energyConsumed
                currentDevice['emissions'] = emission

    tdp1 = multipleDevices["Tesla K80"][0]*currentDevice['TotalTime']/1000
    tdp2 = multipleDevices["Tesla V100-PCIE-16GB"][0]*currentDevice['TotalTime']/1000
    tdp3 = multipleDevices["T4"][0]*currentDevice['TotalTime']/1000

    currentDevice["Tesla K80"] = tdp1
    currentDevice["Tesla V100-PCIE-16GB"] = tdp2
    currentDevice["T4"] = tdp3
    
    print(currentDevice)
    return currentDevice
