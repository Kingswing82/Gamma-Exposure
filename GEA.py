import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

pd.options.display.float_format = '{:,.4f}'.format

# Function to calculate gamma exposure
def calcGammaEx(S, K, vol, T, r, q, OI):
    if T == 0 or vol == 0:
        return 0
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
    return OI * 100 * S * S * 0.01 * gamma 

# Function to check if a date is the third Friday of the month
def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

# Streamlit UI components
st.title("Gamma Exposure Analysis")

# Instructions Section with URL
st.markdown("""
## How to Use the Tool

This tool allows you to analyze options data and calculate gamma exposure. Follow the steps below to use the tool effectively:

1. **Download a CSV file**: From the URL below, ensure the filters for **Volume**, **Expiration Type**, **Options Range** are set to **ALL**. You may select your desired 
    expiration
            
2. **Upload a CSV File**: Click the 'Choose a CSV file' button to upload your options data. The file should contain columns for strike prices, gamma, open interest, and expiration dates.

3. **Inspect Data**: After uploading the file, the tool will automatically parse and display key data points, such as spot price and expiration dates.

4. **Gamma Exposure Analysis**:
    - The tool calculates gamma exposure across different strike prices.
    - It also analyzes the gamma exposure profile, highlighting key levels like the gamma flip point.

5. **Visualizations**:
    - The tool provides visualizations for absolute gamma exposure and gamma exposure by calls and puts.
    - It also charts the gamma exposure profile, showing the potential impact on the market.

6. **Interpret Results**:
    - Use the charts to understand the gamma exposure at various strike prices.
    - Identify critical points such as the spot price and gamma flip point.

7. **Adjust Parameters**:
    - Modify the file or input data and re-upload to see how different scenarios affect gamma exposure.

### Useful Links

- [CBOE SPY Quote Table](https://www.cboe.com/delayed_quotes/spy/quote_table)

## Start by Uploading a CSV File Below:
""")

# File uploader for CSV data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read and parse the CSV file as text to debug
    uploaded_file.seek(0)
    optionsFileData = uploaded_file.read().decode("utf-8").splitlines()

    if len(optionsFileData) < 5:
        st.error("The uploaded file seems to be empty or incorrectly formatted.")
        st.stop()

    # Get Spot Price
    spotLine = optionsFileData[1]
    try:
        spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
    except (IndexError, ValueError):
        st.error("Error parsing the spot price from the file.")
        st.stop()

    fromStrike = 0.8 * spotPrice
    toStrike = 1.2 * spotPrice

    # Get Today's Date
    dateLine = optionsFileData[2]
    try:
        todayDate = dateLine.split('Date: ')[1].strip()

        if " at " in todayDate:
            todayDate = todayDate.split(" at ")[0].strip()
        elif "," in todayDate:
            todayDate = todayDate.split(",")[0].strip()

        todayDate = datetime.strptime(todayDate, '%B %d, %Y')
    except (IndexError, ValueError):
        st.error("Error parsing the date from the file.")
        st.stop()

    # Read the CSV data into a DataFrame
    uploaded_file.seek(0)  # Reset the file pointer before reading as a DataFrame
    try:
        df = pd.read_csv(uploaded_file, sep=",", header=None, skiprows=4, dtype={
            'StrikePrice': float,
            'CallIV': float,
            'PutIV': float,
            'CallGamma': float,
            'PutGamma': float,
            'CallOpenInt': float,
            'PutOpenInt': float
        })
    except pd.errors.EmptyDataError:
        st.error("The CSV file seems to have no data or is incorrectly formatted.")
        st.stop()

    df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                  'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
                  'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y') + timedelta(hours=16)

    # Calculate Spot Gamma
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

    df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9

    # Group by StrikePrice and sum only the numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    dfAgg = df.groupby(['StrikePrice'])[numeric_cols].sum()
    strikes = dfAgg.index.values

    # Plot Absolute Gamma Exposure
    st.subheader("Absolute Gamma Exposure")
    fig, ax = plt.subplots()
    plt.grid()
    plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
    plt.xlim([fromStrike, toStrike])
    chartTitle = f"Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% Move"
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    plt.legend()
    st.pyplot(fig)

    # Plot Gamma Exposure by Calls and Puts
    st.subheader("Gamma Exposure by Calls and Puts")
    fig, ax = plt.subplots()
    plt.grid()
    plt.bar(strikes, dfAgg['CallGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
    plt.bar(strikes, dfAgg['PutGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
    plt.xlim([fromStrike, toStrike])
    chartTitle = f"Total Gamma: ${df['TotalGamma'].sum():.2f} Bn per 1% Move"
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    plt.legend()
    st.pyplot(fig)

    # Calculate and Plot Gamma Exposure Profile
    levels = np.linspace(fromStrike, toStrike, 60)

    df['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 
                        else np.busday_count(todayDate.date(), x.date())/262 for x in df.ExpirationDate]

    nextExpiry = df['ExpirationDate'].min()

    df['IsThirdFriday'] = df['ExpirationDate'].apply(isThirdFriday)

    # Directly filter using the boolean column
    thirdFridays = df.loc[df['IsThirdFriday']]

    nextMonthlyExp = thirdFridays['ExpirationDate'].min()

    totalGamma = []
    totalGammaExNext = []
    totalGammaExFri = []

    # For each spot level, calculate gamma exposure at that point
    for level in levels:
        df['callGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                                              row['daysTillExp'], 0, 0, row['CallOpenInt']), axis=1)

        df['putGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                                             row['daysTillExp'], 0, 0, row['PutOpenInt']), axis=1)    

        totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())

        exNxt = df[df['ExpirationDate'] != nextExpiry]
        totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

        exFri = df[df['ExpirationDate'] != nextMonthlyExp]
        totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

    # Convert to numpy arrays for plotting
    totalGamma = np.array(totalGamma) / 10**9
    totalGammaExNext = np.array(totalGammaExNext) / 10**9
    totalGammaExFri = np.array(totalGammaExFri) / 10**9

    # Find Gamma Flip Point
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]

    # Chart 3: Gamma Exposure Profile
    st.subheader("Gamma Exposure Profile")
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(levels, totalGamma, label="All Expiries")
    plt.plot(levels, totalGammaExNext, label="Ex-Next Expiry")
    plt.plot(levels, totalGammaExFri, label="Ex-Next Monthly Expiry")
    chartTitle = "Gamma Exposure Profile, " + todayDate.strftime('%d %b %Y')
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Index Price', fontweight="bold")
    plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")

    if zeroCrossIdx.size > 0:
        negGamma = totalGamma[zeroCrossIdx]
        posGamma = totalGamma[zeroCrossIdx+1]
        negStrike = levels[zeroCrossIdx]
        posStrike = levels[zeroCrossIdx+1]

        zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
        zeroGamma = zeroGamma[0]
        plt.axvline(x=zeroGamma, color='g', lw=1, label=f"Gamma Flip: {zeroGamma:,.0f}")
        plt.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), facecolor='red', alpha=0.1, transform=ax.get_xaxis_transform())
        plt.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), facecolor='green', alpha=0.1, transform=ax.get_xaxis_transform())

    plt.axhline(y=0, color='grey', lw=1)
    plt.xlim([fromStrike, toStrike])
    plt.legend()
    st.pyplot(fig)
