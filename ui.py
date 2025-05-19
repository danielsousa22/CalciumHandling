from shiny import ui
app_ui = ui.page_fluid(
    ui.h2("Three-Step Peak Analysis"),
    ui.p("Upload a CSV with 'frame' and 'intensity' columns (or two unnamed columns)"),
    ui.input_file("file", "Upload CSV", accept=[".csv"]),
    
    # Step 1: Smoothing Parameters
    ui.page_sidebar(
        ui.sidebar(
            ui.input_numeric("frame_rate", "Frame Rate (Hz)", value=500, min=1),
            ui.input_slider("smooth_window", "Smoothing Window (odd)", 
                          min=5, max=101, value=15, step=2),
            ui.input_numeric("smooth_poly", "Polynomial Order", 
                          value=3, min=1, max=14),
            ui.div(
                ui.input_action_button("update", "Update Graph", class_="btn-primary"),
                ui.output_ui("next_button_1"),
                style="display: flex; gap: 1rem; margin-top: 1rem;"
            ),
            ui.output_text_verbatim("smoothing_status"),
            title="Step 1: Smoothing Parameters",
            width=300
        ),
        ui.output_ui("plot_container_1"),
    ),
    
    # Step 2: Peak Detection
    ui.page_sidebar(
        ui.sidebar(
            ui.h4("Peak Detection Parameters"),
            ui.input_numeric("height", "Height Threshold", value=100),
            ui.input_numeric("distance", "Min Distance (samples)", value=300),
            ui.input_checkbox("skip_first", "Skip First Peak", False),
            ui.input_checkbox("skip_last", "Skip Last Peak", False),
            ui.div(
                ui.input_action_button("back_2", "← Back", class_="btn-warning"),
                ui.input_action_button("next_2", "Next →", class_="btn-success"),
                style="display: flex; gap: 1rem; margin-top: 1rem;"
            ),
            ui.output_text_verbatim("detection_status"),
            title="Step 2: Peak Detection",
            width=300
        ),
        ui.output_plot("detection_plot"),
        ui.output_ui("metrics_output"),
        id="step2_panel",
        class_="conditional-panel"
    ),
    
    # Step 3: Single Peak Analysis
    ui.page_sidebar(
        ui.sidebar(
            ui.h4("Peak Isolation Parameters"),
            ui.output_ui("peak_selector"),
            ui.input_numeric("analysis_start", "Start Time (s)", value=0, min=0),
            ui.input_numeric("analysis_end", "End Time (s)", value=1, min=0.1),
            ui.div(
                ui.input_action_button("back_3", "← Back", class_="btn-warning"),
                ui.input_action_button("analyze", "Analyze Segment", class_="btn-primary"),
                style="display: flex; gap: 1rem; margin-top: 1rem;"
            ),
            ui.output_text_verbatim("analysis_status"),
            title="Step 3: Peak Analysis",
            width=300
        ),
        ui.output_plot("analysis_plot"),
        ui.output_ui("detailed_metrics_output"),
        id="step3_panel",
        class_="conditional-panel"
    ),
    
    ui.tags.style("""
        .conditional-panel {
            display: none;
        }
        .active-panel {
            display: block !important;
        }
        .loading-spinner {
            margin: 20px auto;
            text-align: center;
        }
        .metrics-table {
            margin: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
    """)
)

