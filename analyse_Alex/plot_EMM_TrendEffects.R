plot_EMM_TrendEffects <- function(Data,M,factor_list,factor_group_list,cont_var_list,cont_value_list,facet_list,color_vec,color_pallete_vec = 'black',verbose = FALSE, plot_raw_data = FALSE){
  # load required librarys
  library(dplyr)
  library(emmeans)
  library(ggplot2)
  library(cowplot)
  color_background = '#ebebeb'
  color_violin = '#d4d4d4'
  
  # set up return lists
  plot_list = vector('list',length(factor_list))
  EM_list = vector('list',length(factor_list))
  CEM_list = vector('list',length(factor_list))
  Picture_size_list = vector('list',length(factor_list))
  c = 1
  for(i in seq(1,length(factor_list),1)){
    
    # estimate marginal means
    by_vec = unique(c(factor_list[[i]],setdiff(factor_group_list[[i]],factor_list[[i]])))
    if(length(by_vec) == 0){by_vec = NULL}
    
    temp_cont_value_list = cont_value_list
    var_name_clean <- gsub("^(log|sqrt)\\((.*)\\)$", "\\2", cont_var_list[[i]][1])
    
    temp_cont_value_list[[var_name_clean]]
    temp_cont_value_list[[var_name_clean]] <- unique(Data[[cont_var_list[[i]][1]]])
    #EM=emmeans(M,specs = cont_var_list[[i]][1],by=by_vec,type = "response",at=temp_cont_value_list)
    EM=emmeans(M,specs = var_name_clean,by=by_vec,type = "response",at=temp_cont_value_list)
    
    #include samplesize (n) into dataframe
    DEM = data.frame(confint(EM,calc = c(n = ~.wgt.),adjust='none'))
    #DEM = DEM %>% rename(emmean = rate)
    
    # get the p-values of the trends
    by_vec = unique(c(factor_list[[i]][c(-1)],setdiff(factor_group_list[[i]],factor_list[[i]])))
    if(length(by_vec) == 0){by_vec = NULL}
    EMT = emtrends(M,specs=factor_list[[i]][c(1)],by = by_vec,var = cont_var_list[[i]],infer=TRUE,at=cont_value_list,type = "response")
    DEMT = data.frame(EMT)
    
    # get the contrasts between the trends and append LCL and UCL
    CEM = data.frame(contrast(EMT,interaction="pairwise",infer = TRUE,adjust='none',type = "response"))
    names(CEM)[1] = factor_list[[i]][1] # change first contrast name
    colnames(DEMT)= colnames(CEM)
    CEM = rbind(DEMT,CEM)
    
    # transform cont. var into factor
    #DEM[factor_group_list[[i]]] <- lapply(DEM[factor_group_list[[i]]], as.factor) 
    #CEM[factor_group_list[[i]]] <- lapply(CEM[factor_group_list[[i]]], as.factor)
    if(cont_var_list[[i]] != names(cont_value_list)[1]){
      # calc cutoff from cont_value_list
      edges <- cont_value_list[[1]]
      midpoints <- head(edges, -1) + diff(edges) / 2
      breaks <- c(-Inf, midpoints, Inf)
      print(midpoints)
      print(breaks)
      
      # create dummy factor with the cuttoff value
      Data <- Data %>%
        mutate(dummy_fac = cut(
          .data[[names(cont_value_list)[1]]],
          breaks = breaks,
          labels = edges,
          right = FALSE,
          include.lowest = TRUE
        ),dummy_fac = factor(dummy_fac, levels = edges),
        # overwrite cont values with dummy_fac
        !!names(cont_value_list)[1] := dummy_fac
        ) %>% dplyr::select(-dummy_fac)
      
      #Data[factor_group_list[[i]]] <- lapply(Data[factor_group_list[[i]]], as.factor)
    }
    Data[factor_group_list[[i]]] <- lapply(Data[factor_group_list[[i]]], as.factor)
    
    # calc 2 way interaction, if specified in factor-list
    if(length(factor_list[[i]]) >= 2){
      # estimate marginal means
      by_vec = unique(c(factor_list[[i]][c(-1,-2)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EMT = emtrends(M,specs = factor_list[[i]][c(1,2)],by=by_vec,var = cont_var_list[[i]][1],infer=TRUE,at=cont_value_list)
      # get the contrasts of contrasts and append LCL and UCL
      CCEM = data.frame(contrast(EMT,interaction = 'pairwise',infer = TRUE,adjust='none'))
      
      # combine with first level contrasts
      colnames(CCEM) = colnames(CEM)
      CEM = rbind(CEM,CCEM)
      
      # get the p-values of the trends
      by_vec = unique(c(factor_list[[i]][c(-2)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EMT = emtrends(M,specs = factor_list[[i]][c(2)],by = by_vec,var = cont_var_list[[i]][1],infer=TRUE,at=cont_value_list)
      print(EMT)
      temp = data.frame(contrast(EMT,interaction = "pairwise",infer = TRUE,adjust='none'))
      # switch column 1 and 2, in order to match with CEM dataframe
      temp = temp %>% dplyr::select(all_of(contains(factor_list[[i]])),everything())
      colnames(temp) = colnames(CEM)
      
      # combine data frames
      CEM = rbind(CEM,temp)
    }
    
    # calc 3 way interaction, if specified in factor-list
    if(length(factor_list[[i]]) >= 3){
      # estimate marginal means
      by_vec = unique(c(factor_list[[i]][c(-1,-2,-3)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EMT = emtrends(M,specs = factor_list[[i]][c(1,2,3)],by=by_vec,var = cont_var_list[[i]][1],infer=TRUE,at=cont_value_list)
      # get the contrasts of contrasts of contrasts and append LCL and UCL
      CCCEM = data.frame(contrast(EMT,interaction = 'pairwise',infer = TRUE,adjust='none'))
      
      # combine with global data frame
      colnames(CCCEM) = colnames(CEM)
      CEM = rbind(CEM,CCCEM)
      
      # get the p-values of the trends
      by_vec = unique(c(factor_list[[i]][c(-3)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EMT = emtrends(M,specs = factor_list[[i]][c(3)],by = by_vec,var = cont_var_list[[i]][1],infer=TRUE,at=cont_value_list)
      temp = data.frame(contrast(EMT,interaction = "pairwise",infer = TRUE,adjust='none'))
      # restore original column order
      temp = temp %>% dplyr::select(all_of(contains(factor_list[[i]])),everything())
      colnames(temp) = colnames(CEM)
      
      # combine data frames
      CEM = rbind(CEM,temp)
      print(CEM)
      
      # get the p-values of the trends
      by_vec = unique(c(factor_list[[i]][c(-1,-3)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EMT = emtrends(M,specs = factor_list[[i]][c(1,3)],by = by_vec,var = cont_var_list[[i]][1],infer=TRUE,at=cont_value_list)
      temp = data.frame(contrast(EMT,interaction = "pairwise",infer = TRUE,adjust='none'))
      # restore original column order
      temp = temp %>% dplyr::select(all_of(contains(factor_list[[i]])),everything())
      colnames(temp) = colnames(CEM)
      
      # combine data frames
      CEM = rbind(CEM,temp)
      print(CEM)
      
      # get the p-values of the trends
      by_vec = unique(c(factor_list[[i]][c(-2,-3)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EMT = emtrends(M,specs = factor_list[[i]][c(2,3)],by = by_vec,var = cont_var_list[[i]][1],infer=TRUE,at=cont_value_list)
      temp = data.frame(contrast(EMT,interaction = "pairwise",infer = TRUE,adjust='none'))
      # restore original column order
      temp = temp %>% dplyr::select(all_of(contains(factor_list[[i]])),everything())
      colnames(temp) = colnames(CEM)
      
      # combine data frames
      CEM = rbind(CEM,temp)
      print(CEM)
    }
    if(verbose == TRUE){
      print('###################################################################')
      print('Marginal-Mean-Estimates:')
      print(DEM)
      print(' ')
      print('###################################################################')
      print('Contrasts:')
      print(CEM)
    }
    
    
    
    DEM[factor_group_list[[i]]] <- lapply(DEM[factor_group_list[[i]]], as.factor) 
    CEM[factor_group_list[[i]]] <- lapply(CEM[factor_group_list[[i]]], as.factor)
    # plot all calculated contrasts ==> effects
    pos_dodge = position_dodge(width = 1)
    if(length(facet_list[[i]]) == 2){
      facet_string = as.formula(paste0(facet_list[[i]][2],'~',facet_list[[i]][1]))
    }else{
      if(length(facet_list[[i]]) == 1){
        facet_string = as.formula(paste0('.~',facet_list[[i]][1]))
      }
    }
    # 1. Automatisch richtige Spalte für die Effekt-Schätzung erkennen
    possible_estimate_names <- c("emmean", "estimate", "rate", "ratio", "prob")
    aes_string_var_name <- intersect(possible_estimate_names, names(CEM))[1]
    #levels(CEM$Block) = asas.character(cont_value_list[unlist(cont_var_list)])
    
    m_effects <- ggplot(data = CEM,aes_string(x = aes_string_var_name,y = factor_list[[i]][1],color = color_vec[i],linetype=var_name_clean)) +
      geom_point(position = pos_dodge) +
      geom_errorbar(aes_string(x = aes_string_var_name,xmin = 'lower.CL',xmax = 'upper.CL'),width = 0.02,position = pos_dodge) +
      geom_vline(xintercept = 0,linetype = "dashed",color = 'black') +
      #facet_grid(facet_string,scales = 'fixed') +
      scale_x_continuous(position="bottom") +
      ylab(paste0(as.character(formula(M))[2])) +
      #xlab("effects (ms)")+ylab("trend/contrast") +
      theme(legend.title = element_blank(),legend.position = "top",legend.direction = "horizontal",legend.box = "horizontal") + guides(color = guide_legend(nrow = 1)) +
      geom_text(aes(label=paste0('p=',round(p.value,3))),vjust = -0.2,position = pos_dodge,show.legend = F) +
      scale_color_manual(values=color_pallete_vec) +
      theme(panel.background = element_rect(fill = color_background,colour = color_background))
    # === Facet nur anwenden, wenn facet_string existiert ===
    if(length(facet_list[[i]]) == 2 | length(facet_list[[i]]) == 1){
      m_effects <- m_effects + facet_grid(facet_string, scales = 'fixed')
    }
    
    
    # calc a useful pic size for plotting
    facet_vars <- ggplot_build(m_effects)$layout$layout
    facet_row = max(facet_vars$ROW)
    facet_col = max(facet_vars$COL)
    gb <- ggplot_build(m_effects)
    y_data <- gb$data[[1]]$y
    n_y = length(unique(y_data))
    effects_width = 5*facet_col
    effects_height = max(c(1.5*n_y*facet_row,10))
    
    # plotting the results
    possible_estimate_names <- c("emmean", "estimate", "rate", "ratio", "prob")
    aes_string_var_name <- intersect(possible_estimate_names, names(DEM))[1]
    #levels(DEM$Block) = as.character(seq(1,12,1))
    
    m_response <- ggplot(data = DEM,aes_string(x=var_name_clean,y=aes_string_var_name,color=color_vec[i],group=factor_list[[i]][1])) + 
      geom_line(linewidth = 0.7,position = pos_dodge) +
      geom_point(position = pos_dodge) +
      geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.2,position = pos_dodge) +
      ylab(paste0(as.character(formula(M))[2])) +
      theme(legend.title = element_blank(),legend.position = "none",legend.direction = "horizontal",legend.box = "horizontal") + guides(color = guide_legend(nrow = 1)) +
      #facet_grid(facet_string,scales='fixed',shrink = TRUE) +
      scale_color_manual(values=color_pallete_vec) +
      theme(panel.background = element_rect(fill = color_background,colour = color_background))
    # === Facet nur anwenden, wenn facet_string existiert ===
    if(length(facet_list[[i]]) == 2 | length(facet_list[[i]]) == 1){
      m_response <- m_response + facet_grid(facet_string, scales = 'fixed',shrink = TRUE)
    }
    if(plot_raw_data == TRUE){
      pos_jitter_raw <- position_dodge(width = 1)
      m_response <- m_response +
        stat_summary(data = Data,aes_string(x = var_name_clean,y = as.character(formula(M))[2],color = color_vec[i]),fun.data = mean_se,geom = "errorbar",width = 0.1,position = pos_dodge,alpha=0.3) +
        stat_summary(data = Data,aes_string(x = var_name_clean,y = as.character(formula(M))[2],color = color_vec[i]),fun = mean,geom = "point",size = 2,position = pos_dodge,alpha=0.3) +
        stat_summary(data = Data,aes_string(x = var_name_clean,y = as.character(formula(M))[2], color = color_vec[i]),fun = mean,geom = "line",linewidth = 0.5,position = pos_dodge,alpha=0.3)
    }
    
    # calc a useful pic size for plotting
    facet_vars <- ggplot_build(m_response)$layout$layout
    facet_row = max(facet_vars$ROW)
    facet_col = max(facet_vars$COL)
    gb <- ggplot_build(m_response)
    x_data <- gb$data[[1]]$x
    n_x = length(unique(x_data))
    response_width = n_x*facet_col
    response_height = max(c(5*facet_row,10))
    
    # arange the EMM plots wtih the effect plots
    plot_list[[c]] = plot_grid(m_effects, m_response, nrow = 2, ncol = 1, align = "v", axis = "lr", labels = "AUTO", scale = 1, rel_heights = c(1, 1))
    EM_list[[c]] = DEM
    CEM_list[[c]] = CEM
    Picture_size_list[[c]] = c(max(effects_width,response_width),effects_height + response_height)
    c = c+1
  }
  # set the return values
  return(list(
    plots = plot_list,
    contrasts = CEM_list,
    estimates = EM_list,
    pic_sizes = Picture_size_list
  ))
}

