import plotly.graph_objects as go


def plot_positions(true_positions, estimated_positions):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=true_positions[:, 0], y=true_positions[:, 1], z=true_positions[:, 2],
        mode='lines+markers',
        name='True Position',
        marker=dict(size=4),
        line=dict(color='blue', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter3d(
        x=estimated_positions[:, 0], y=estimated_positions[:,
                                                           1], z=estimated_positions[:, 2],
        mode='lines+markers',
        name='Estimated Position',
        marker=dict(size=4),
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title="True vs Estimated Position in 3D",
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position"
        )
    )
    fig.show()


def plot_errors(errors):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(errors) + 1)),
        y=errors,
        mode='lines+markers',
        name='Estimation Error',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title="Estimation Error over Time",
        xaxis_title="Time Step",
        yaxis_title="Euclidean Error"
    )
    fig.show()
