{
  "steps": [
    {
      "init": {
        "assign": [
          {
            "input": "Enter a string in Æon:"
          },
          {
            "output": ""
          }
        ]
      }
    },
    {
      "convert": {
        "for": {
          "value": "i",
          "range": [
            0,
            {
              "length": "${input}"
            }
          ],
          "steps": [
            {
              "assign": [
                {
                  "c": "${input[i]}"
                }
              ]
            },
            {
              "switch": {
                "switch": "${c}",
                "cases": [
                  {
                    "value": "a",
                    "assign": [
                      {
                        "output": "${output}𐐁"
                      }
                    ]
                  },
                  {
                    "value": "b",
                    "assign": [
                      {
                        "output": "${output}𐐂"
                      }
                    ]
                  },
                  {
                    "value": "c",
                    "assign": [
                      {
                        "output": "${output}𐐃"
                      }
                    ]
                  },
                  {
                    "value": "d",
                    "assign": [
                      {
                        "output": "${output}𐐄"
                      }
                    ]
                  },
                  {
                    "value": "e",
                    "assign": [
                      {
                        "output": "${output}𐐅"
                      }
                    ]
                  },
                  {
                    "value": "f",
                    "assign": [
                      {
                        "output": "${output}𐐆"
                      }
                    ]
                  },
                  {
                    "value": "g",
                    "assign": [
                      {
                        "output": "${output}𐐇"
                      }
                    ]
                  },
                  {
                    "value": "h",
                    "assign": [
                      {
                        "output": "${output}𐐈"
                      }
                    ]
                  },
                  {
                    "value": "i",
                    "assign": [
                      {
                        "output": "${output}𐐉"
                      }
                    ]
                  },
                  {
                    "value": "j",
                    "assign": [
                      {
                        "output": "${output}𐐊"
                      }
                    ]
                  },
                  {
                    "value": "k",
                    "assign": [
                      {
                        "output": "${output}𐐋"
                      }
                    ]
                  },
                  {
                    "value": "l",
                    "assign": [
                      {
                        "output": "${output}𐐌"
                      }
                    ]
                  },
                  {
                    "value": "m",
                    "assign": [
                      {
                        "output": "${output}𐐍"
                      }
                    ]
                  },
                  {
                    "value": "n",
                    "assign": [
                      {
                        "output": "${output}𐐎"
                      }
                    ]
                  },
                  {
                    "value": "o",
                    "assign": [
                      {
                        "output": "${output}𐐏"
                      }
                    ]
                  },
                  {
                    "value": "p",
                    "assign": [
                      {
                        "output": "${output}𐐐"
                      }
                    ]
                  },
                  {
                    "value": "q",
                    "assign": [
                      {
                        "output": "${output}𐐑"
                      }
                    ]
                  },
                  {
                    "value": "r",
                    "assign": [
                      {
                        "output": "${output}𐐒"
                      }
                    ]
                  },
                  {
                    "value": "s",
                    "assign": [
                      {
                        "output": "${output}𐐓"
                      }
                    ]
                  },
                  {
                    "value": "t",
                    "assign": [
                      {
                        "output": "${output}𐐔"
                      }
                    ]
                    
