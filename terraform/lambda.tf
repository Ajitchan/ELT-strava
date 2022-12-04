resource "aws_lambda_function" "strava_lf" {
  filename = ""
  function_name = "strava_lf"
  handler = "strava.lambda_handler"
  role = ""
  runtime = ""
  timeout = "300"
}