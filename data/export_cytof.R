# Export the Levine_13dim mass cytometry benchmark (Weber & Robinson 2016) from the
# Bioconductor package HDCytoData to CSV.
#
# Exported: the cell-type ("type") marker intensities, untransformed
# (the Python loader applies arcsinh with cofactor 5), and the manually gated
# population label (population_id; "unassigned" = cells outside every gate).
#
# Usage (from project root):  Rscript data/export_cytof.R
# Outputs: data/cytof/Levine_13dim.csv.gz, data/cytof/Levine_13dim_populations.csv

suppressPackageStartupMessages({
  library(HDCytoData)
  library(SummarizedExperiment)
})

out <- file.path("data", "cytof")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

export <- function(name, se) {
  is_type <- colData(se)$marker_class == "type"
  X <- assay(se, "exprs")[, is_type, drop = FALSE]
  pop <- as.character(rowData(se)$population_id)
  df <- data.frame(X, population = pop, check.names = FALSE)
  con <- gzfile(file.path(out, paste0(name, ".csv.gz")), "w")
  write.csv(df, con, row.names = FALSE)
  close(con)
  tab <- as.data.frame(table(population = pop))
  tab$share <- tab$Freq / sum(tab$Freq)
  write.csv(tab[order(tab$Freq), ], file.path(out, paste0(name, "_populations.csv")), row.names = FALSE)
  cat(sprintf("%s: %d cells, %d type markers, %d gated populations\n",
              name, nrow(X), ncol(X), length(setdiff(unique(pop), "unassigned"))))
}

export("Levine_13dim", Levine_13dim_SE())
