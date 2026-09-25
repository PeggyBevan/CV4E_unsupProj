#plot correlation of variables
cor_plot <- function(data, method=c("pearson", "kendall", "spearman"),
                     sort=FALSE,
                     axis_text_size=12,
                     number_text_size=3,
                     legend=FALSE){
  method <- match.arg(method)
  index <- sapply(data, is.numeric)
  qdata <- data[index]
  qdata <- na.omit(qdata)
  # bind global variables to keep check from warning
  r <- stats::cor(qdata, method=method)
  p <- ggcorrplot(r,
                  hc.order = sort,
                  colors = c("blue", "white", "red"),
                  type = "lower",
                  lab = TRUE,
                  lab_size=number_text_size,
                  show.legend=legend)
  n <- format(nrow(qdata), big.mark=",")
  if (method == "pearson"){
    subtitle <- paste0("Pearson correlations (n = ",n, ")")
  }
  if (method == "spearman"){
    subtitle <- paste0("Spearman rank order correlations (n = ",n, ")")
  }
  if (method == "kendall"){
    subtitle <- paste0("Kendall rank order correlations (n = ",n, ")")
  }
  p <- p + labs(title = "Correlation Matrix",
                subtitle = subtitle) +
    theme(axis.text.x=element_text(size=axis_text_size),
          axis.text.y=element_text(size=axis_text_size),
          plot.subtitle = element_text(size=8,
                                       face="plain"))
  return(p)
}
