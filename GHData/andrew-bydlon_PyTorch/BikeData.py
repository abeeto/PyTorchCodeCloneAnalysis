import numpy, torch, csv, pandas

bikes = numpy.loadtxt("Bike-Sharing-Dataset/hour.csv", dtype=numpy.float16, delimiter=',', skiprows=1, converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes).cuda()

print(bikes)

dailyBikes = bikes.view(-1, 24, bikes.shape[1])
dailyBikes = dailyBikes.transpose(1,2)

day1 = bikes[:24].long()
weatherOneHot = torch.zeros(day1.shape[0], 4).cuda()
day1[:,9]
weatherOneHot.scatter_(dim=1, index=day1[:,9].unsqueeze(1) - 1, value=1.0).to(dtype=torch.float16)

torch.cat((bikes[:24], weatherOneHot.to(dtype=torch.float16)), 1)[:1]

dailyWeatherOneHot = torch.zeros(dailyBikes.shape[0], 4, dailyBikes.shape[2]).cuda()
dailyWeatherOneHot.shape
dailyWeatherOneHot.scatter_(1, dailyBikes[:,9,:].long().unsqueeze(1)-1, 1.0)

dailyBikes = torch.cat((dailyBikes, dailyWeatherOneHot.to(dtype=torch.float16)), dim=1)
dailyBikes[:, 9, :] = (dailyBikes[:, 9, :] - 1)/3

from Standardize import standardizeColumn

standardizeColumn(dailyBikes, 10)
print(dailyBikes)