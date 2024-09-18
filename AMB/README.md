# Active Memory Benchmark (AMB)

The Active Memory Benchmark (AMB) is composed of more than 20K of highly challenging visual queries from EPIC-KITCHENS. These queries cover different levels of video reasoning (sequencing, concurrency and temporal grounding) to assess detailed video understanding capabilities. The benchmark consists of 8 different types of queries, each represented in separate JSON files. Each JSON file contains questions related to video data, and answers can be visual or textual

<center>
<figure>
    <img src="../assets/benchmark.png" width="900px" />
</figure>
</center>

## Data Format

Each JSON file contains a list of questions with the following fields:

- **id**: Unique identifier for the question.
- **video_id**: Identifier of the video from the EPIC-KITCHENS dataset.
- **question**: The textual question being asked.
- **question_image**: Applicable only for questions that include a visual crop. Contains frame number and crop dimensions. Format: `{'object': [[frame, [x0, y0, x1, y1]], ...]}`.
- **answers**: Contains five possible answers, which can be sequences of object crops, sets of crops, or text.
- **correct**: The id of the correct answer.

### Example

```json
[
    {
        "id": "Q1_000100",
        "video_id": "P01_01",
        "question": "What is the correct sequence of objects with which the fridge is interacted?",
        "question_image": {
            "fridge": [
                [938, [0, 0, 1, 1]],
                [1515, [0, 0, 1, 1]],
                [1114, [0, 0, 1, 1]]
            ]
        },
        "answers": "answers": [
            "{'1': {'container': [[64532, [0.44, 0.63, 0.067, 0.98]], [93383, [0.24, 0.45, 0.46, 0.80]], [64838, [0.55, 0.64, 0.74, 0.97]]]},
              '2': {'carrot': [[4814, [0.45, 0.46, 0.57, 0.65]], [4534, [0.46, 0.46, 0.56, 0.64]], [35404, [0.51, 0.38, 0.61, 0.50]]]}, 
            // other 3 answers with the same format format
        ],,
        "correct": 5
    }
]
```

### Fields Description

- **id**: Identifier for each question (e.g., `Q1_000100`).
- **video_id**: The specific video from EPIC-KITCHENS dataset (e.g., `P01_01`).
- **question**: The question asked about the video.
- **question_image**: Visual crop details (if applicable). Contains frames and bounding boxes for crops.
- **answers**: List of potential answers, including object sequences, crops, or text.
- **correct**: The index of the correct answer in the `answers` list.

Each object can have 1 to 3 crops, depending on the availability of ground truths.

## Query Types

The benchmark includes 8 different types of queries, each saved in its respective JSON file. Ensure to refer to each file for specific query types and their formats.