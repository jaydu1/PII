library(clusterProfiler)
library(org.Hs.eg.db)
library(AnnotationDbi)
library(rrvgo)

library(purrr)
library(stringr)
library(ggplot2)
library(patchwork)
library(dplyr)

path_rs <- '~/batch_correction/results/LUHMES'
setwd(path_rs)
df <- read.csv(sprintf('%s/res_tmp.csv', path_rs), row.names=1)





plot_GO <- function(genes_to_test, filename_GO, filename_GO_tree){
    GO_results <- enrichGO(gene = genes_to_test, OrgDb = org.Hs.eg.db, keyType = "SYMBOL", ont = "BP",
                           # pvalueCutoff  = 0.1
    )
    
    pdf(filename_GO, width = 5, height = 6)
    plot(barplot(GO_results, showCategory = 10))
    dev.off()
    
    go_analysis <- as.data.frame(GO_results)
    simMatrix <- calculateSimMatrix(go_analysis$ID,
                                    orgdb="org.Hs.eg.db",
                                    ont="BP",
                                    method="Rel")
    scores <- setNames(-log10(go_analysis$qvalue), go_analysis$ID)
    reducedTerms <- reduceSimMatrix(simMatrix,
                                    scores,
                                    threshold=0.7,
                                    orgdb="org.Hs.eg.db")
    
    pdf(filename_GO_tree, width = 8, height = 4)
    treemapPlot(reducedTerms)
    dev.off()
}

for(i in c(1:3)){
name <- colnames(df)[i]
genes_to_test <- rownames(df[df[,name]=='True',])
plot_GO(genes_to_test, sprintf("%sGO_%s.pdf", path_rs, name), sprintf("%sGO_treemap_%s.pdf", path_rs, name))
}

name <- colnames(df)[1]
genes_to_test <- rownames(df[(df[,name]=='True')|(df[,'common']=='True'),])
plot_GO(genes_to_test, sprintf("%sGO_%s_all.pdf", path_rs, name), sprintf("%sGO_treemap_%s_all.pdf", path_rs, name))