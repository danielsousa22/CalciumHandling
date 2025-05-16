# app.R
library(shiny)
library(reticulate)

# Tell reticulate where your Python script lives
use_python("/usr/bin/python3", required = TRUE)  # adjust path as needed
source_python("peak_analysis_app.py")

ui <- fluidPage(
  titlePanel("Interactive Peak Analysis"),
  sidebarLayout(
    sidebarPanel(
      fileInput("csvfile", "Upload CSV",
                accept = c(".csv")),
      numericInput("frame_rate", "Frame Rate", value = 500),
      sliderInput("smooth_window", "Smoothing Window",
                  min = 5, max = 101, value = 15, step = 2),
      numericInput("smooth_poly", "Smoothing Polyorder", value = 3, min = 1),
      hr(),
      h4("Peak Detection"),
      numericInput("height", "Height threshold", value = 100),
      numericInput("distance", "Min peak distance", value = 300),
      checkboxInput("skip_first", "Skip first peak", FALSE),
      checkboxInput("skip_last",  "Skip last peak", FALSE),
      actionButton("detect", "Run Peak Detection"),
      hr(),
      h4("Fine-tuning"),
      numericInput("before_sec", "Seconds before peak", value = 0),
      numericInput("after_sec",  "Seconds after peak", value = 0),
      actionButton("tune", "Run Fine-Tuning"),
      hr(),
      actionButton("compute", "Compute Metrics & Download")
    ),
    mainPanel(
      plotOutput("peakPlot"),
      plotOutput("windowPlot"),
      verbatimTextOutput("metrics"),
      downloadButton("downloadMetrics", "Download Metrics CSV")
    )
  )
)

server <- function(input, output, session) {
  # Reactive values to store intermediate results
  vals <- reactiveValues(
    signal = NULL,
    time   = NULL,
    peaks  = NULL,
    avg_bpm = NULL,
    local_info = NULL,
    metrics_df = NULL
  )
  
  observeEvent(input$csvfile, {
    req(input$csvfile)
    df <- read.csv(input$csvfile$datapath)
    vals$time   <- df$frame / input$frame_rate
    vals$signal <- savgol_filter(df$intensity,
                                 window_length = input$smooth_window,
                                 polyorder     = input$smooth_poly)
  })
  
  observeEvent(input$detect, {
    req(vals$signal)
    # call Python peak detector
    out <- detect_peaks_with_skip(
      signal    = vals$signal,
      height    = input$height,
      distance  = input$distance,
      skip_first = input$skip_first,
      skip_last = input$skip_last
    )
    vals$peaks <- out[[1]]
    # plot in R
    output$peakPlot <- renderPlot({
      plot(vals$time, vals$signal, type = "l", main = "Detected Peaks",
           xlab = "Time (s)", ylab = "Intensity")
      points(vals$time[vals$peaks], vals$signal[vals$peaks], col = "red", pch = 19)
    })
    # compute avg bpm
    bpm <- py$calculate_bpm(vals$peaks, input$frame_rate)
    vals$avg_bpm <- bpm[[2]]
  })
  
  observeEvent(input$tune, {
    req(vals$peaks, vals$avg_bpm)
    info <- detect_first_local_peak(
      signal    = vals$signal,
      peak_indices = vals$peaks,
      avg_bpm   = vals$avg_bpm,
      frame_rate = input$frame_rate,
      before_sec = input$before_sec,
      after_sec  = input$after_sec
    )
    vals$local_info <- info
    output$windowPlot <- renderPlot({
      wr <- info$window_range
      plot((wr[1]:wr[2]) / input$frame_rate,
           vals$signal[wr[1]:wr[2]], type = "l", main = "Local Window")
      abline(v = info$selected_peak / input$frame_rate, col = "blue", lty = 2)
    })
  })
  
  observeEvent(input$compute, {
    req(vals$local_info)
    sel <- vals$local_info$selected_peak
    # compute all metrics
    baseline <- compute_baseline(vals$signal)
    amplitude <- compute_amplitude(vals$signal, baseline, sel)
    t_peak <- compute_time_to_peak(vals$time, sel)
    decay  <- fit_signal_decay(vals$time, vals$signal, sel)
    decay_times <- compute_decay_times(vals$time, vals$signal,
                                       sel, decay$tau)
    # assemble data.frame
    dfm <- data.frame(
      metric = c("baseline", "amplitude", "time_to_peak", "tau",
                 paste0("decay_", c("80","50","10"))),
      value  = c(baseline, amplitude, t_peak, decay$tau,
                 unlist(decay_times))
    )
    vals$metrics_df <- dfm
    output$metrics <- renderPrint({ print(dfm) })
  })
  
  output$downloadMetrics <- downloadHandler(
    filename = function() {
      paste0(tools::file_path_sans_ext(input$csvfile$name), "_metrics.csv")
    },
    content = function(file) {
      write.csv(vals$metrics_df, file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)
