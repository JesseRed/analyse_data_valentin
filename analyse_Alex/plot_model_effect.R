plot_model_effect <- function(baseData, best_model, target_vars, n_points = 10) {
  all_preds <- all.vars(formula(best_model))[-1]
  
  missing_vars <- setdiff(target_vars, all_preds)
  if (length(missing_vars) > 0) {
    stop(paste("The following target_vars are not in the model:", paste(missing_vars, collapse = ", ")))
  }
  
  filling_vars <- setdiff(all_preds, target_vars)
  
  target_list <- lapply(target_vars, function(var) {
    x <- baseData[[var]]
    if (is.factor(x) || is.character(x) || is.ordered(x)) {
      if (!is.factor(x)) x <- factor(x)
      levels(x)
    } else if (is.numeric(x) || is.integer(x)) {
      seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = n_points)
    } else {
      stop(paste("Unsupported variable type for", var, ":", class(x)))
    }
  })
  names(target_list) <- target_vars
  newdat <- expand.grid(target_list)
  
  for (var in filling_vars) {
    x <- baseData[[var]]
    if (is.factor(x) || is.character(x) || is.ordered(x)) {
      if (!is.factor(x)) x <- factor(x)
      newdat[[var]] <- levels(x)[1]
    } else if (is.numeric(x) || is.integer(x)) {
      newdat[[var]] <- mean(x, na.rm = TRUE)
    } else {
      stop(paste("Unsupported variable type for", var, ":", class(x)))
    }
  }
  
  newdat$fit <- predict(best_model, newdata = newdat)
  
  x_var <- baseData[[target_vars[1]]]
  is_x_numeric <- is.numeric(x_var) || is.integer(x_var)
  
  p <- ggplot(baseData, aes_string(
    x = target_vars[1],
    y = as.character(formula(best_model))[2]
  ))
  
  if (length(target_vars) == 4) {
    if (is_x_numeric) {
      p <- p +
        geom_point(aes_string(shape = target_vars[2], linetype = target_vars[3], color = target_vars[4]), size = 2, alpha = 0.7) +
        geom_line(data = newdat, aes_string(x = target_vars[1], y = "fit", color = target_vars[4], linetype = target_vars[3]), linewidth = 1)
    } else {
      p <- p +
        geom_boxplot(aes_string(fill = target_vars[4]), outlier.shape = NA, alpha = 0.6) +
        geom_jitter(color = "black", width = 0.2, alpha = 0.3, size = 1.5) +
        geom_point(data = newdat, aes_string(x = target_vars[1], y = "fit", color = target_vars[4], shape = target_vars[2]), size = 3)
    }
    p <- p + facet_grid(as.formula(paste(target_vars[3], "~", target_vars[2])))
    
  } else if (length(target_vars) == 3) {
    if (is_x_numeric) {
      p <- p +
        geom_point(aes_string(shape = target_vars[2], color = target_vars[3]), size = 2, alpha = 0.7) +
        geom_line(data = newdat, aes_string(x = target_vars[1], y = "fit", color = target_vars[3]), linewidth = 1)
    } else {
      p <- p +
        geom_boxplot(aes_string(fill = target_vars[3]), outlier.shape = NA, alpha = 0.6) +
        geom_jitter(color = "black", width = 0.2, alpha = 0.3, size = 1.5) +
        geom_point(data = newdat, aes_string(x = target_vars[1], y = "fit", color = target_vars[3]), size = 3)
    }
    p <- p + facet_grid(as.formula(paste(target_vars[3], "~", target_vars[2])))
    
  } else if (length(target_vars) == 2) {
    if (is_x_numeric) {
      p <- p +
        geom_point(aes_string(color = target_vars[2]), size = 2, alpha = 0.7) +
        geom_line(data = newdat, aes_string(x = target_vars[1], y = "fit", color = target_vars[2]), linewidth = 1)
    } else {
      p <- p +
        geom_boxplot(aes_string(fill = target_vars[2]), outlier.shape = NA, alpha = 0.6) +
        geom_jitter(color = "black", width = 0.2, alpha = 0.3, size = 1.5) +
        geom_point(data = newdat, aes_string(x = target_vars[1], y = "fit", color = target_vars[2]), size = 3)
    }
  } else {
    stop("Function currently supports 2, 3, or 4 target_vars")
  }
  
  p <- p +
    labs(
      x = target_vars[1],
      y = as.character(formula(best_model))[2],
      title = paste("Effect of", target_vars[1]),
      subtitle = "boxplots/jitter = data, lines/points = model prediction"
    ) +
    #theme_minimal(base_size = 14) +
    theme(legend.position = "bottom")
  
  return(p)
}
