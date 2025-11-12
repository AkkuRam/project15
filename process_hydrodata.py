import pandas as pd
from functools import partial

# Order of degress of freedom
DOF_MAP = {
    'Surge': 1,
    'Sway': 2,
    'Heave': 3,
    'Roll': 4, 
    'Pitch': 5, 
    'Yaw': 6
}

# Ensuring consistent naming convention
COLUMN_MAP = {
    'wave_direction': 'dir',
    'omega': 'w',
    'complex': 'i',
    'radiating_dof': 'row',
    'influenced_dof': 'col',
    'added_mass': 'A',
    'radiation_damping': 'B',
    'excitation_force': 'F'
}

def pivot(df: pd.DataFrame, col: str, re: bool) -> pd.DataFrame:
    # Data in df should contain the same components as [3]
    return (
        df[df['i'] == ('re' if re else 'im')] # Extraction of real and imag compoennt
        [['col', 'row', col]] # Extraction of relevant column
        .pivot(columns='col', index='row', values=col) # Creating pivot table for rectangular data
        # Ensuring the columns and rows are in the correct dof order
        .sort_index(key=lambda x: x.map(DOF_MAP)).transpose()
        .sort_index(key=lambda x: x.map(DOF_MAP)).transpose()
    )

def extract_A(df: pd.DataFrame) -> pd.DataFrame:
    # For added mass we do not consider directional term
    (_, df) = next(iter(df.groupby('dir')))
    assert df is not None
    # Extracting added mass matrix
    return df.groupby('w', group_keys=True).apply(partial(pivot, col='A', re=True))

def extract_B(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # For damping we do not consider directional term
    (_, df) = next(iter(df.groupby('dir')))
    return df.groupby('w', group_keys=True).apply(partial(pivot, col='B', re=True))

def extract_F(df: pd.DataFrame) -> pd.DataFrame:
    # For force directional term is required
    return (
        df.groupby(['w', 'dir']).apply(lambda x: pivot(x, 'F', True).mean(axis=0)),
        df.groupby(['w', 'dir']).apply(lambda x: pivot(x, 'F', False).mean(axis=0))
    )

if __name__ == "__main__":
    PATH = lambda x: f'resources/{x}'
    # Loading df and renaming to used naming convention
    df = pd.read_csv(PATH('diffraction_data.csv')).rename(columns=COLUMN_MAP)

    # Extracting relevant coefficients
    df_A = extract_A(df)
    print("Writing A matrix to csv...")
    df_A.to_csv(PATH('A.csv'))
    df_B = extract_B(df)
    print("Writing B matrix to csv...")
    df_B.to_csv(PATH('B.csv'))
    print("Writing F matrix to csv...")
    (df_F_r, df_F_i) = extract_F(df)
    df_F_r.to_csv(PATH('F_real.csv'))
    df_F_i.to_csv(PATH('/F_complex.csv'))
