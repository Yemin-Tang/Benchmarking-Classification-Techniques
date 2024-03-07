library(Benchmarking)
library(readxl)

# read the data
data <- read_excel("C:/Users/Ariff Farhan Rahdi/Desktop/Benchmarking Techniques/BCC/ourdataset 1 (1).xlsx")

# input & output selection
x <- with(data, cbind(total_debt_to_assets, liabilities_to_assets))
y <- matrix(data = c(data$current_assets_to_liabilities, data$net_income_to_assets, data$cash_flows_to_assets, data$working_capital_to_assets, data$retained_earning_to_assets, data$operating_earning_to_assets, data$sales_to_total_asset), ncol = 7)

# calculating efficiency
# variable returns to scale
bcc <- dea(x,y, RTS="vrs", ORIENTATION="in")
print(bcc)

eff(bcc)
data.frame(bcc$eff)
summary(bcc)
sl <- slack(x,y, bcc)
data.frame(eff(bcc),eff(sl),sl$slack,sl$sx,sl$sy,lambda(sl))

dea.plot(x,y, RTS="vrs", ORIENTATION = "in-out")
dea.plot.frontier(x,y, txt=1:dim(x[1]))

# bootstrap dea
bccb <- dea.boot(x,y, NREP=3000, RTS="vrs", ORIENTATION = "in", alpha = 0.05) # to have a good bootstrap
bccb

# calculating super efficiency
superbcc <- sdea(x,y, RTS="vrs", ORIENTATION="in")
superbcc
print(peers(superbcc,NAMES = TRUE),quote = FALSE)

# excess input compared to frontier input
excess(bcc,x)
x-eff(bcc)*x