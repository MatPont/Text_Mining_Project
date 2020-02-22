setwd(dirname(rstudioapi::getSourceEditorContext()$path))

tp <- c(10025, 1155568, 3905450)
tn <- c(8132098, 6245665, 1025225)
fp <- c(4250529, 3104986, 355104)
fn <- c(15019, 1901452, 7121892)

all_min <- min(tp, tn, fp, fn)
all_max <- max(tp, tn, fp, fn)

plot(tp, type="b", ylim = c(all_min, all_max), xaxt="n", ylab = "", xlab = "Alpha")
lines(tn, pch = 2, lty = 4, type="b")
lines(fp, pch = 3, type="b")
lines(fn, pch = 4, lty = 4, type="b")
axis(1, at=1:3, labels=c(0.9, 0.75, 0.5))
legend("top",legend=c("TP", "TN", "FP", "FN"), lty = c(1,4,1,4), pch = 1:4, cex = 0.75)
