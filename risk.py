print("=" * 50)
print("EMOTIONAL RISK PREDICTION")
print("=" * 50)

try:

    # Read masking score
    with open("masking_result.txt", "r", encoding="utf-8") as file:

        score = int(file.read().strip().split(" ")[-1])

    # Check risk level
    if score <= 30:

        risk = "LOW"

    elif score <= 70:

        risk = "MEDIUM"

    else:

        risk = "HIGH"

    # Display result
    print("\n-----------------------------")
    print("Masking Score :", score, "%")
    print("Risk Level    :", risk)

    # Save risk result
    with open("risk_result.txt", "w", encoding="utf-8") as file:

        file.write(risk)

    print("\nRisk result saved successfully!")

except FileNotFoundError:

    print("masking_result.txt not found.")

except ValueError:

    print("Invalid masking score.")

except Exception as e:

    print("Error:", e)

print("Program Closed")