import streamlit as st
import pathlib
import jupedsim as jps
import pedpy
from numpy.random import normal
from shapely import Polygon
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
import matplotlib.pyplot as plt

# Streamlit UI for parameter selection
st.title("Pedestrian Dynamics Simulation")
st.sidebar.header("Simulation Settings")

# User inputs for simulation parameters
num_agents = st.sidebar.slider("Number of Agents", min_value=5, max_value=50, value=20, step=5)
agent_speed_mean = st.sidebar.slider("Mean Agent Speed (m/s)", min_value=1.0, max_value=2.0, value=1.34, step=0.05)
agent_speed_std_dev = st.sidebar.slider("Speed Standard Deviation", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

# Define the walkable and spawning areas
area = Polygon([(0, 0), (12, 0), (12, 12), (10, 12), (10, 2), (0, 2)])
walkable_area = pedpy.WalkableArea(area)
spawning_area = Polygon([(0, 0), (6, 0), (6, 2), (0, 2)])
exit_area = Polygon([(10, 11), (12, 11), (12, 12), (10, 12)])

# Distribute agents within the spawning area
pos_in_spawning_area = jps.distributions.distribute_by_number(
    polygon=spawning_area,
    number_of_agents=num_agents,
    distance_to_agents=0.4,
    distance_to_polygon=0.2,
    seed=1,
)

# Plot initial configuration using matplotlib
def plot_initial_configuration(walkable_area, spawning_area, starting_positions, exit_area):
    fig, ax = plt.subplots()
    ax = pedpy.plot_walkable_area(walkable_area=walkable_area)
    ax.fill(*spawning_area.exterior.xy, color="lightgrey")
    ax.fill(*exit_area.exterior.xy, color="indianred")
    ax.scatter(*zip(*starting_positions))
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.set_aspect("equal")
    return fig

# Display initial configuration plot
st.subheader("Initial Configuration")
initial_fig = plot_initial_configuration(walkable_area, spawning_area, pos_in_spawning_area, exit_area)
st.pyplot(initial_fig)

# Run simulation and display animation
if st.button("Run Simulation"):
    trajectory_file = "corner.sqlite"  # Output file
    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=area,
        trajectory_writer=jps.SqliteTrajectoryWriter(output_file=pathlib.Path(trajectory_file))
    )

    exit_id = simulation.add_exit_stage(exit_area.exterior.coords[:-1])
    journey = jps.JourneyDescription([exit_id])
    journey_id = simulation.add_journey(journey)

    # Assign agent parameters based on user input
    v_distribution = normal(agent_speed_mean, agent_speed_std_dev, num_agents)
    for pos, v0 in zip(pos_in_spawning_area, v_distribution):
        simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                journey_id=journey_id, stage_id=exit_id, position=pos, v0=v0
            )
        )

    # Run the simulation
    while simulation.agent_count() > 0:
        simulation.iterate()

    # Load and create animation
    trajectory_data, walkable_area = read_sqlite_file(trajectory_file)
    animation_figure = animate(trajectory_data, walkable_area, every_nth_frame=5)

    # Display animation with Plotly
    st.subheader("Simulation Animation")
    st.plotly_chart(animation_figure)
