plot_EMM_Effects <- function(Data,M,factor_list,factor_group_list,cont_value_list,facet_list,color_vec,color_pallete_vec = 'black',verbose = FALSE, plot_raw_data = FALSE){
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
  a = 1
  for(i in seq(1,length(factor_list),1)){
    # create formula string for first level contrasts
    # determines the x-axis labels
    
    #formula_string = paste0('~ ',factor_list[[i]][1],'|',paste(factor_list[[i]][-1],collapse = '+'))
    # estimate marginal means
    #formula_string = as.formula(paste0(formula_string,'+',paste(names(cont_value_list),collapse = '+')))
    #EM=emmeans(M,formula_string,at=cont_value_list)
    
    # estimate marginal means
    by_vec = unique(c(factor_list[[i]][c(-1)],setdiff(factor_group_list[[i]],factor_list[[i]])))
    if(length(by_vec) == 0){by_vec = NULL}
    EM=emmeans(M,specs = factor_list[[i]][c(1)],by=by_vec,at=cont_value_list,type='response')
    
    #include samplesize (n) into dataframe
    DEM = data.frame(confint(EM,calc = c(n = ~.wgt.),adjust='none'))
    
    # get the contrasts and append LCL and UCL
    CEM = data.frame(contrast(EM,interaction="pairwise",infer = TRUE,adjust='none'))
    names(CEM)[1] = 'contrast' # change first contrast name
    
    # transform cont. var into factor
    DEM[names(cont_value_list)] <- lapply(DEM[names(cont_value_list)], as.factor) 
    CEM[names(cont_value_list)] <- lapply(CEM[names(cont_value_list)], as.factor)
    if(length(cont_value_list) > 0){
      # calc cutoff from cont_value_list
      edges <- cont_value_list[[1]]
      midpoints <- head(edges, -1) + diff(edges) / 2
      breaks <- c(-Inf, midpoints, Inf)
      
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
      
      # Dynamisches Filtern:
      Data <- Data %>%
        filter(
          !!!lapply(names(cont_value_list), function(col) {
            vals <- cont_value_list[[col]]
            expr(!!sym(col) %in% !!vals)
          })
        )
    }
    # calc 2 way interaction, if specified in factor-list
    if(length(factor_list[[i]]) >= 2){
      # create formula string for second level contrasts
      #formula_string = as.formula(paste0('~',factor_list[[i]][1],'+',factor_list[[i]][2],'|',paste(factor_list[[i]][c(-1,-2)],collapse = '+')))
      # estimate marginal means
      #EM=emmeans(M,formula_string)

      # estimate marginal means
      by_vec = unique(c(factor_list[[i]][c(-1,-2)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EM=emmeans(M,specs = factor_list[[i]][c(1,2)],by=by_vec,at=cont_value_list,type='response')
      
      # get the contrasts and append LCL and UCL
      # contrasts of contrasts
      CCEM = data.frame(contrast(EM,interaction = 'pairwise',infer = TRUE,adjust='none'))
      
      # combine with first level contrasts
      colnames(CCEM) = colnames(CEM)
      CEM = rbind(CEM,CCEM)
    }
    
    # calc 3 way interaction, if specified in factor-list
    if(length(factor_list[[i]]) >= 3){
      # create formula string for second level contrasts, but with third factor
      #formula_string = as.formula(paste0('~',factor_list[[i]][1],'+',factor_list[[i]][3],'|',paste(factor_list[[i]][c(-1,-3)],collapse = '+')))
      # estimate marginal means
      #EM=emmeans(M,formula_string)
      
      # estimate marginal means
      by_vec = unique(c(factor_list[[i]][c(-1,-3)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EM=emmeans(M,specs = factor_list[[i]][c(1,3)],by=unique(by_vec),at=cont_value_list,type='response')

      # get the contrasts and append LCL and UCL
      # contrasts of contrasts 
      CCEM = data.frame(contrast(EM,interaction = 'pairwise',infer = TRUE,adjust='none'))
      
      # switch column 2 and 3, in order to match with CEM dataframe
      CCEM = CCEM %>% dplyr::select(all_of(contains(factor_list[[i]])),everything())
      colnames(CCEM) = colnames(CEM)
      # combine with other contrasts
      CEM = rbind(CEM,CCEM)
      
      # create formula string for second level contrasts, but with third factor
      #formula_string = as.formula(paste0('~',factor_list[[i]][1],'+',factor_list[[i]][2],'+',factor_list[[i]][3]))
      # estimate marginal means
      #EM=emmeans(M,formula_string)
      
      by_vec = unique(c(factor_list[[i]][c(-1,-2,-3)],setdiff(factor_group_list[[i]],factor_list[[i]])))
      if(length(by_vec) == 0){by_vec = NULL}
      EM=emmeans(M,specs = factor_list[[i]][c(1,2,3)],by=unique(by_vec),at=cont_value_list,type='response')
  
      # get the contrasts and append LCL and UCL
      # contrasts of contrasts of contrasts
      CCCEM = data.frame(contrast(EM,interaction = 'pairwise',infer = TRUE,adjust='none'))
      
      # combine with all other contrasts
      colnames(CCCEM) = colnames(CEM)
      CEM = rbind(CEM,CCCEM)
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
    # Hole die Link-Funktion aus dem Modell
    model_link <- family(M)$link
    # Bestimme die passende Intercept-Referenzlinie
    xintercept_value <- switch(model_link,
                               "log" = 1,
                               "logit" = 0,
                               "identity" = 0,
                               0  # Default für alle anderen Link-Funktionen
    )
    m_effects <- ggplot(data = CEM,aes_string(x = aes_string_var_name,y = 'contrast',color = color_vec[i])) +
      geom_point(position = pos_dodge) +
      geom_errorbar(aes_string(x = aes_string_var_name,xmin = 'lower.CL',xmax = 'upper.CL'),width = 0.02,position = pos_dodge) +
      geom_vline(xintercept = xintercept_value,linetype = "dashed",color = 'black') +
      facet_grid(facet_string,scales = 'fixed') +
      scale_x_continuous(position="bottom") +
      #xlab("effects (ms)")+ylab("contrast") +
      theme(legend.title = element_blank(),legend.position = "top",legend.direction = "horizontal",legend.box = "horizontal") + guides(color = guide_legend(nrow = 1)) +
      geom_text(aes(label=paste0('p=',round(p.value,3))),vjust = -0.2,position = pos_dodge,show.legend = F) +
      scale_color_manual(values=color_pallete_vec) +
      theme(panel.background = element_rect(fill = color_background,colour = color_background)) + 
      theme(axis.text.y = element_text(angle = 45, hjust = 1))
    
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
    
    m_response <- ggplot(DEM,aes_string(x=factor_list[[i]][1],y=aes_string_var_name, color = color_vec[i]))
    if(plot_raw_data == TRUE){
      library(ggrepel)
      m_response = m_response + 
        stat_summary(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2],group = color_vec[i]),data=Data,color = 'black',fun.y=mean, geom="point", shape=23, size=3,position = pos_dodge,alpha=0.75) +
        geom_violin(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2]),data=Data,fill='grey',position = pos_dodge,alpha=0.3) +
        geom_jitter(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2]),data=Data,position=position_jitterdodge(jitter.width = 0.2, dodge.width = 1))
        #geom_text_repel(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2],label='ID'),data = Data, size = 3, max.overlaps = 15)  # ID-Label
    }
    m_response = m_response +
      geom_point(shape=16,size=4,position = pos_dodge) +
      geom_line(aes_string(group = color_vec[i]),position = pos_dodge) +
      geom_errorbar(data=DEM,aes_string(x=factor_list[[i]][1],ymin='lower.CL',ymax='upper.CL'),width=0.2,position = pos_dodge) +
      ylab(paste0(as.character(formula(M))[2],' (ms)')) +
      theme(legend.title = element_blank(),legend.position = "none") +
      facet_grid(facet_string,scales='fixed',shrink = TRUE) +
      scale_color_manual(values=color_pallete_vec) +
      theme(panel.background = element_rect(fill = color_background,colour = color_background))

    # calc a useful pic size for plotting
    facet_vars <- ggplot_build(m_response)$layout$layout
    facet_row = max(facet_vars$ROW)
    facet_col = max(facet_vars$COL)
    gb <- ggplot_build(m_response)
    x_data <- gb$data[[1]]$x
    n_x = length(unique(x_data))
    response_width = 5*n_x*facet_col
    response_height = max(c(5*facet_row,10))
    
    # arange the EMM plots wtih the effect plots
    plot_list[[a]] = plot_grid(m_effects, m_response, nrow = 2, ncol = 1, align = "v", axis = "lr", labels = "AUTO", scale = 1, rel_heights = c(1, 1))
    EM_list[[a]] = DEM
    CEM_list[[a]] = CEM
    Picture_size_list[[a]] = c(max(effects_width,response_width),effects_height + response_height)
    a = a+1
  }
  
  # ##### test #####
  # library(ggsignif)
  # group_cols <- factor_group_list[[1]]
  # 
  # CEM_filtered <- CEM %>%
  #   filter(Reduce(`&`, lapply(group_cols, function(col) CEM[[col]] %in% levels(DEM[[col]]))))
  # 
  # CEM_filtered_neg <- CEM %>%
  #   filter(!Reduce(`&`, lapply(group_cols, function(col) CEM[[col]] %in% levels(DEM[[col]]))))
  # 
  # possible_estimate_names <- c("emmean", "estimate", "rate", "ratio", "prob")
  # aes_string_var_name <- intersect(possible_estimate_names, names(CEM))[1]
  # # Hole die Link-Funktion aus dem Modell
  # model_link <- family(M)$link
  # 
  # m_effects_filtered <- ggplot(data = CEM_filtered_neg,aes_string(x = aes_string_var_name,y = 'contrast')) +
  #   geom_point(position = pos_dodge) +
  #   geom_errorbar(aes_string(x = aes_string_var_name,xmin = 'lower.CL',xmax = 'upper.CL'),width = 0.02,position = pos_dodge) +
  #   geom_vline(xintercept = xintercept_value,linetype = "dashed",color = 'black') +
  #   facet_wrap(facet_string,scales = 'fixed') +
  #   scale_x_continuous(position="bottom") +
  #   #xlab("effects (ms)")+ylab("contrast") +
  #   theme(legend.title = element_blank(),legend.position = "top",legend.direction = "horizontal",legend.box = "horizontal") + guides(color = guide_legend(nrow = 1)) +
  #   geom_text(aes(label=paste0('p=',round(p.value,3))),vjust = -0.2,position = pos_dodge,show.legend = F) +
  #   scale_color_manual(values=color_pallete_vec) +
  #   theme(panel.background = element_rect(fill = color_background,colour = color_background)) + 
  #   theme(axis.text.y = element_text(angle = 45, hjust = 1))
  # 
  # 
  # # Levels manuell definieren (falls nötig)
  # levels_VR <- levels(DEM[[factor_list[[1]][1]]])
  # levels_MTime <- levels(DEM[[factor_group_list[[1]][1]]])
  # 
  # # 2️⃣ Hilfsfunktion (gleich wie vorher)
  # extract_comparison <- function(contrast_string) {
  #   strsplit(as.character(contrast_string), " - ")[[1]]
  # }
  # # Funktion für die Position innerhalb des Dodges
  # get_dodge_pos <- function(vr_game, mtime, dodge_width = 1) {
  #   vr_game = '1'
  #   mtime = '2'
  #   x_base <- which(levels_MTime == mtime)
  #   n_groups <- length(levels_VR)
  #   group_index <- which(levels_VR == vr_game)
  #   shift <- ((group_index - 1) - (n_groups - 1)/2) * dodge_width / n_groups
  #   return(x_base + shift)
  # }
  # 
  # # 3️⃣ CEM vorbereiten
  # CEM_signif <- CEM_filtered %>%
  #   rowwise() %>%
  #   mutate(
  #     contrast_chr = as.character(contrast),
  #     group1 = extract_comparison(contrast_chr)[1],
  #     group2 = extract_comparison(contrast_chr)[2],
  #     xmin = get_dodge_pos(group1, !!sym(factor_group_list[[1]][1])),
  #     xmax = get_dodge_pos(group2, !!sym(factor_group_list[[1]][1])),
  #     annotations = ifelse(p.value < 0.001, "***",
  #                          ifelse(p.value < 0.01, "**",
  #                                 ifelse(p.value < 0.05, "*",
  #                                        paste0("p=", round(p.value, 3)))))
  #   ) %>%
  #   ungroup()
  # 
  # # 4️⃣ Einmalige Y-Position pro Kontrast bestimmen
  # y_max <- max(Data[[as.character(formula(M))[2]]], na.rm = TRUE)
  # unique_contrasts <- unique(CEM_signif$contrast_chr)
  # 
  # # z. B. gleichmäßig gestaffelt über den höchsten Punkt
  # y_positions <- y_max + seq(0.05, by = 1, length.out = length(unique_contrasts))
  # 
  # # y-Position pro Kontrast zuordnen (damit pro Facet gleich bleibt)
  # y_map <- data.frame(contrast_chr = unique_contrasts,
  #                     y_position = y_positions)
  # 
  # CEM_signif <- left_join(CEM_signif, y_map, by = "contrast_chr")
  # 
  # # plotting the results
  # possible_estimate_names <- c("emmean", "estimate", "rate", "ratio", "prob")
  # aes_string_var_name <- intersect(possible_estimate_names, names(DEM))[1]
  # 
  # m_response <- ggplot(DEM,aes_string(x=factor_list[[i]][1],y=aes_string_var_name, color = factor_list[[i]][2]))
  # #if(plot_raw_data == TRUE){
  #   library(ggrepel)
  #   m_response = m_response + 
  #     stat_summary(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2],group = color_vec[i]),data=Data,color = 'black',fun.y=mean, geom="point", shape=23, size=3,position = pos_dodge,alpha=0.75) +
  #     geom_violin(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2]),data=Data,fill='grey',position = pos_dodge,alpha=0.3) +
  #     geom_jitter(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2]),data=Data,position=position_jitterdodge(jitter.width = 0.2, dodge.width = 1)) 
  #   # geom_text_repel(aes_string(x=factor_list[[i]][1],y=as.character(formula(M))[2],label='ID'),data = Data, size = 3, max.overlaps = 15)  # ID-Label
  # #}
  # m_response = m_response +
  #   geom_point(shape=16,size=4,position = pos_dodge) +
  #   #geom_line(aes_string(group = color_vec[i]),position = pos_dodge) +
  #   geom_errorbar(data=DEM,aes_string(ymin='lower.CL',ymax='upper.CL'),width=0.2,position = pos_dodge) +
  #   ylab(paste0(as.character(formula(M))[2])) +
  #   theme(legend.title = element_blank(),legend.position = "bottom") +
  #   #facet_grid(facet_string,scales='fixed',shrink = TRUE) +
  #   scale_color_manual(values=color_pallete_vec) +
  #   theme(panel.background = element_rect(fill = color_background,colour = color_background))
  # 
  # # 5️⃣ Signifikanzlinien zeichnen (jetzt mit fixer Höhe je Kontrast)
  # m_response_signif <- m_response +
  #   geom_signif(
  #     data = CEM_signif,
  #     aes(
  #       xmin = xmin,
  #       xmax = xmax,
  #       annotations = annotations,
  #       y_position = y_position
  #     ),
  #     manual = TRUE,
  #     tip_length = 0.02,
  #     textsize = 4,
  #     color = "black"
  #   )
  # 
  # plot_grid(m_effects_filtered, m_response_signif, nrow = 2, ncol = 1, align = "v", axis = "lr", labels = c("A",''), scale = 1, rel_heights = c(0.5, 1))
  # # test ends
  
  
  
  
  # set the return values
  return(list(
    plots = plot_list,
    contrasts = CEM_list,
    estimates = EM_list,
    pic_sizes = Picture_size_list
  ))
}

