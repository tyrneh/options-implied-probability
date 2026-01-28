from oipd.data_access import sources
try:
    dates = sources.list_expiry_dates("GME")
    if not dates:
        print("No expiries found")
        exit()
    expiry = dates[0]
    chain, snap = sources.fetch_chain("GME", expiries=[expiry])
    print("Columns:", chain.columns.tolist())
    print("First row:", chain.iloc[0].to_dict())
except Exception as e:
    print(e)
