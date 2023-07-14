import pickle
import streamlit as st
import pandas as pd

# Load weights
if 'linreg' not in st.session_state:
    st.session_state['linreg'] = {
        'Seattle': pickle.load(open('weights/LR_Seattle.sav', 'rb')),
        'Boston': pickle.load(open('weights/LR_Boston.sav', 'rb'))
    }
    
if 'rfreg' not in st.session_state:
    st.session_state['rfreg'] = {
        'Seattle': pickle.load(open('weights/RF_Seattle.sav', 'rb')),
        'Boston': pickle.load(open('weights/RF_Boston.sav', 'rb'))
    }

if 'scaler' not in st.session_state:
    st.session_state['scaler'] = {
        'Seattle': pickle.load(open('weights/Seattle_scaler.pkl', 'rb')),
        'Boston': pickle.load(open('weights/Boston_scaler.pkl', 'rb'))
    }

st.title('Group 2 - Demo')
st.title('AirBNB Prices Prediction')

st.markdown('## Choosing fundamental:', unsafe_allow_html=True)
room_col = st.columns(2)
with room_col[0]:
    bedrooms = st.number_input('Number of bedrooms', step=1, min_value=0)
    st.write('Number of bedrooms', bedrooms)
    
with room_col[1]:
    bathrooms = st.number_input('Number of bathrooms', step=1, min_value=0)
    st.write('Number of bathrooms', bathrooms)

st.markdown('## Choosing type of room and property:', unsafe_allow_html=True)
type_col = st.columns(2)
with type_col[0]:
    # - Loại phòng (1 trong 3 loại, yn)
    room_type = st.selectbox(
        'Room type',
        ('Entire home apartment', 'Private room', 'Shared room'))
    st.write('Room type:', room_type)

with type_col[1]:
    # - Loại căn hộ (1 trong 6 loại, yn)
    property_type = st.selectbox(
        'Propery type',
        ('Bed and Breakfast', 'Condominium', 
        'House', 'Loft', 'Other (Cabin, Boat, Dorm ...)', 'Townhouse'))
    st.write('Room type:', property_type)

st.markdown('## Choosing furniture:', unsafe_allow_html=True)
choice_col = st.columns(3)
with choice_col[0]:
    tv = st.checkbox('TV?')
    if tv:
        st.write('Has TV')
        tv_ = 1
    else:
        st.write('No TV')
        tv_ = 0

    # "elevator"
    elevator = st.checkbox('Elevator?')
    if elevator:
        st.write('Has elevator')
        elevator_ = 1
    else:
        st.write('No elevator')
        elevator_ = 0

with choice_col[1]:
    # "gym"
    gym = st.checkbox('Gym?')
    if gym:
        st.write('Has gym')
        gym_ = 1
    else:
        st.write('No gym')
        gym_ = 0

    # "hot_tub_sauna_or_pool"
    hot_tub_sauna_or_pool = st.checkbox('Pool?')
    if hot_tub_sauna_or_pool:
        st.write('Has Pool')
        hot_tub_sauna_or_pool_ = 1
    else:
        st.write('No pool')
        hot_tub_sauna_or_pool_ = 0

with choice_col[2]:
    # "internet"
    internet = st.checkbox('Internet?')
    if internet:
        st.write('Has Internet')
        internet_ = 1
    else:
        st.write('No Internet')
        internet_ = 0

    # "pets_allowed"
    pets_allowed = st.checkbox('Pet Allowed?')
    if pets_allowed:
        st.write('Allowing pets')
        pets_allowed_ = 1
    else:
        st.write('Not allowing pet')
        pets_allowed_ = 0

if st.button('Submit'):
    room = [0, 0, 0]
    if room_type == 'Entire home apartment':
        room[0] = 1
    elif room_type == 'Private room':
        room[1] = 1
    else:
        room[2] = 1
    
    property = [0, 0, 0, 0, 0, 0]
    if property_type == 'Bed and Breakfast':
        property[0] = 1
    elif room_type == 'Condominium':
        property[1] = 1
    elif room_type == 'House':
        property[2] = 1
    elif room_type == 'Loft':
        property[3] = 1
    elif room_type == 'Other':
        property[4] = 1
    else:
        property[5] = 1

    list_input = [bedrooms, bathrooms, tv_,
                  elevator_, gym_, hot_tub_sauna_or_pool_,
                  internet_, pets_allowed_] + room + property
    # scaled
    Seattle_scaled = st.session_state['scaler']['Seattle'].transform([list_input])
    Boston_scaled = st.session_state['scaler']['Boston'].transform([list_input])

    # Results
    lr = {
        'Seattle': st.session_state['linreg']['Seattle'].predict(Seattle_scaled)[0][0],
        'Boston': st.session_state['linreg']['Boston'].predict(Boston_scaled)[0][0]
    }
    rf = {
        'Seattle': st.session_state['rfreg']['Seattle'].predict(Seattle_scaled)[0],
        'Boston': st.session_state['rfreg']['Boston'].predict(Boston_scaled)[0]
    }

    st.markdown('### Price', unsafe_allow_html=True)
    df = pd.DataFrame(
        index=['Linear regression', 'Random forest regression'],
        data=[
            lr, rf
        ]
    )
    df['Seattle'] = df['Seattle'].map('${:,.2f}'.format)
    df['Boston'] = df['Boston'].map('${:,.2f}'.format)
    # st.dataframe(df.T, width=20)
    st.table(df)
