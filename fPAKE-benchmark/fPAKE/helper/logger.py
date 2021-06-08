import asyncio
import csv
import json

class logger:

    def __init__(self, deviceName):
        wfile = open("benchmark"+deviceName+".json","w")
        self.csvWriter = csv.writer(wfile)

    def setIteration(self,iter):
        self.iteration = iter


    async def write(self, data):
        self.csvWriter.writerow((self.iteration,data))

    def log(self, data):
        asyncio.run(self.write(data))
