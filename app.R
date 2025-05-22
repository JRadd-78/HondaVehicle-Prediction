
# Load required libraries
library(tidyverse)
library(ranger)
library(shiny)
library(bslib)

# Load the cleaned dataset
sales_data <- read_csv("Sales_Data.csv")

# Extract actual levels from the dataset
age_levels <- levels(factor(sales_data$AgeRange_Clean))
income_levels <- levels(factor(sales_data$IncomeLevel_Clean))
gender_levels <- levels(factor(sales_data$Gender))
zip_levels <- levels(factor(sales_data$ADJZipCode))

# Ensure formatting
sales_data <- sales_data %>%
  mutate(
    VehicleModel = as.factor(VehicleModel),
    ADJZipCode = as.factor(ADJZipCode),
    SellingPrice = as.numeric(SellingPrice),
    AgeRange_Clean = factor(AgeRange_Clean, levels = age_levels),
    IncomeLevel_Clean = factor(IncomeLevel_Clean, levels = income_levels),
    Gender = factor(Gender, levels = gender_levels)
  ) %>%
  drop_na()

# Load pre-trained model
rf_model <- readRDS("bootstrap_forest_model.rds")

# Define UI
ui <- fluidPage(
  theme = bs_theme(
    bootswatch = "flatly",
    primary = "#0073C2",
    base_font = "sans-serif"
  ),

  tags$head(
    tags$style(HTML("
      body {
        background: linear-gradient(to right, #f0f4f8, #dceefb);
      }
      .main-container {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-top: 30px;
      }
      .progress {
        height: 30px;
      }
      .progress-bar {
        font-size: 16px;
        font-weight: bold;
        line-height: 30px;
      }
    "))
  ),

  div(style = "text-align: center; padding-top: 20px;",
      tags$img(src = "https://img1.wsimg.com/blobby/go/72e115c3-a759-4da3-a901-ced48b652c56/downloads/f8492d45-cda7-4fcd-bba3-1c0a54a826d6/14%20MOH%20NEW%20LOGO%20H%20K%2B285.png?ver=1738832769439", height = "100px"),
      h1("Car Model Prediction", style = "color: #0073C2; padding-top: 10px;")
  ),

  div(class = "main-container",
      sidebarLayout(
        sidebarPanel(
          tags$h4("Input Customer Details"),
          numericInput("selling_price", "Customer Budget:", value = 25000, min = 5000, max = 100000, step = 1000),
          selectInput("adj_zip_code", "ZIP Code:", choices = zip_levels),
          selectInput("age_range", "Age Range:", choices = age_levels),
          selectInput("income_level", "Income Level:", choices = income_levels),
          selectInput("gender", "Gender:", choices = gender_levels),
          actionButton("predict", "🚀 Predict Model", class = "btn btn-primary btn-lg"),
          br(),
          tags$p("Fill in the details and click Predict to get the top 3 recommended vehicle models.", style = "color: #7f8c8d;")
        ),
        mainPanel(
          div(style = "text-align: center;",
              h3("Top 3 Predicted Vehicle Models:"),
              uiOutput("top_predictions_ui"),
              tags$hr(),
              tags$p("These predictions are based on customer demographics and preferences.", style = "color: #7f8c8d;")
          )
        )
      )
  )
)

# Define Server
server <- function(input, output) {
  prediction <- eventReactive(input$predict, {
    tryCatch({
      new_data <- data.frame(
        ADJZipCode = factor(input$adj_zip_code, levels = levels(sales_data$ADJZipCode)),
        SellingPrice = as.numeric(input$selling_price),
        AgeRange_Clean = factor(input$age_range, levels = levels(sales_data$AgeRange_Clean)),
        IncomeLevel_Clean = factor(input$income_level, levels = levels(sales_data$IncomeLevel_Clean)),
        Gender = factor(input$gender, levels = levels(sales_data$Gender))
      )

      pred_probs <- predict(rf_model, new_data)$predictions
      top_3 <- sort(pred_probs[1, ], decreasing = TRUE)[1:3]
      data.frame(
        Model = names(top_3),
        Probability = round(top_3 * 100, 1)
      )
    }, error = function(e) {
      print(paste("Prediction error:", e$message))
      return(data.frame(Model = "Prediction failed", Probability = 0))
    })
  })

  output$top_predictions_ui <- renderUI({
    preds <- prediction()

    if (nrow(preds) == 0 || all(is.na(preds$Probability))) {
      return(h4("No predictions available."))
    }

    tagList(
      lapply(1:nrow(preds), function(i) {
        percent <- preds$Probability[i]
        label <- preds$Model[i]

        color_class <- if (percent >= 80) {
          "bg-success"
        } else if (percent >= 50) {
          "bg-warning"
        } else {
          "bg-danger"
        }

        div(
          tags$h4(label),
          div(class = "progress",
              div(class = paste("progress-bar progress-bar-striped progress-bar-animated", color_class),
                  role = "progressbar",
                  style = paste0("width: ", percent, "%;"),
                  paste0(percent, "%")
              )
          ),
          br()
        )
      })
    )
  })
}

# Run the app
shinyApp(ui = ui, server = server)
