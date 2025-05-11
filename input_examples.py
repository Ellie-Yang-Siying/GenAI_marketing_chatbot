# input_examples.py

# Primary input_example used for log_model()
input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is the total spend for 2024 on Goldfish on search campaigns?"
        }
    ]
}

# Additional test examples
input_examples = [
    {
        "messages": [
            {"role": "user", "content": "How does the total spend compare to 2023?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How much did we spend in Halloween in 2024 and what was the overall ROI?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Which search terms delivered the best ROI?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Which search term had the best ROAS in the month of April?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Which keywords delivered the highest impressions during back to school?"}
        ]
    }
]

# Merge input_example at the front
input_examples = [input_example] + input_examples
