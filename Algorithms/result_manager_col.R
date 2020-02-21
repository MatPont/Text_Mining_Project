setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(ggplot2)
library(svglite)

best_result_path <- "../Results/best/"

datasets <- c("classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb")
algos <- c("CoclustInfo", "CoclustMod", "CoclustSpecMod")

plot_grouped_barplot <- function(data){
  x_names <- data[,1]
  Score <- data[,2]
  value <- as.double(data[,3])
  data <- data.frame(data)  
  my_barplot <- ggplot(data, aes(fill=Score, y=value, x=x_names)) + xlab("Algorithm") + ylab("Score") +
    geom_bar(position="dodge", stat="identity") + ylim(0, 1.0) + theme_grey(base_size = 15)
  print(my_barplot)
  return(my_barplot)
}

for(dataset in datasets){
  cat("##############\n#", dataset, "\n##############\n")
  words_path <- paste(best_result_path, dataset, "_words", sep = "")
  res_algos <- c()
  for(algo in algos){
    algo_res_col_path <- paste(best_result_path, dataset, "_", algo, "_col.txt", sep = "")
    algo_res_col <- mean(as.matrix(read.table(algo_res_col_path, sep=","))[,1])
    
    algo_res_row_path <- paste(best_result_path, "row_results/", dataset, "_", algo, ".txt", sep = "")
    algo_res_row <- mean(as.matrix(read.table(algo_res_row_path, sep=",")))
    
    res_algos <- rbind(res_algos, c(algo, "Row Score", algo_res_row))
    res_algos <- rbind(res_algos, c(algo, "Column Score", algo_res_col))
  }
  print(res_algos)
  svg_file_name <- paste(best_result_path, dataset, ".svg", sep="")
  # svg(filename = svg_file_name)
  # plot_grouped_barplot(res_algos)
  # dev.off()
  my_barplot <- plot_grouped_barplot(res_algos)
  print(my_barplot)
  break
  #ggsave(svg_file_name, device = "svg")
}

# create a dataset
specie <- c(rep("sorgho" , 3) , rep("poacee" , 3) , rep("banana" , 3) , rep("triticum" , 3) )
condition <- rep(c("normal" , "stress" , "Nitrogen") , 4)
value <- abs(rnorm(12 , 0 , 15))
data <- data.frame(specie,condition,value)

# Grouped
ggplot(data, aes(fill=condition, y=value, x=specie)) + 
  geom_bar(position="dodge", stat="identity")
