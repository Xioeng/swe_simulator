import os
from typing import List, Tuple

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .io import load_bathymetry_and_meshgrid, read_solutions


def normalize_velocities_for_plotting(
    v_x: np.ndarray, v_y: np.ndarray, max_arrow_length: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize velocity vectors to limit maximum arrow length."""
    velocity_magnitude = np.sqrt(v_x**2 + v_y**2)
    scale = np.ones_like(velocity_magnitude)
    mask = velocity_magnitude > max_arrow_length
    scale[mask] = max_arrow_length / velocity_magnitude[mask]
    v_x_scaled = v_x * scale
    v_y_scaled = v_y * scale
    return v_x_scaled, v_y_scaled


def initialize_plot(output_path: str, **kargs) -> Tuple[plt.Figure, plt.Axes]:
    """Initialize a Matplotlib plot with Cartopy for Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects.
    """
    bathymetry, (X, Y) = load_bathymetry_and_meshgrid(output_path)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8, 14))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
    # Add satellite imagery using Google Maps tiles
    google_tiles = cimgt.GoogleTiles(style="street")
    ax.add_image(
        google_tiles,
        14,  # zoom level, adjust as needed
        interpolation="bilinear",
    )
    return fig, ax


def plot_solution(output_path: str, frame: int = 0, **kargs) -> None:
    """Plot the solution at a given frame from Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    frame : int
        Frame number to plot.
    """

    result = read_solutions(output_path, frames_list=[frame])
    bathymetry = result["bathymetry"]
    X, Y = result["meshgrid"]
    sol = result["solutions"][0]

    h = sol[0, :, :]  # assuming depth is the first equation
    wave_treshold = kargs.get("wave_treshold", 1e-2)
    free_surface = bathymetry + h
    free_surface[h < wave_treshold] = np.nan
    v_x, v_y = sol[1, :, :] / h, sol[2, :, :] / h  # velocities

    v_x[h <= wave_treshold] = 0.0
    v_y[h <= wave_treshold] = 0.0

    max_arrow_length = kargs.get("max_arrow_length", 0.5)
    v_x_scaled, v_y_scaled = normalize_velocities_for_plotting(
        v_x, v_y, max_arrow_length
    )

    velocity = np.sqrt(v_x**2 + v_y**2)
    plt.style.use("dark_background")
    h_fig = plt.figure(figsize=(8, 14))
    # Add a satellital image using cartopy's Stamen imagery
    h_ax = h_fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    h_ax.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
    # Add satellite imagery using Google Maps tiles

    google_tiles = cimgt.GoogleTiles(style="street")
    h_ax.add_image(
        google_tiles,
        14,  # zoom level, adjust as needed
        interpolation="bilinear",
    )
    contour = h_ax.contourf(
        X, Y, free_surface, levels=50, cmap="viridis", alpha=0.7
    )  # perceptually uniform, good for scalar fields
    velocity_field = h_ax.quiver(
        X,
        Y,
        v_x_scaled,
        v_y_scaled,
        velocity,
        angles="xy",
        cmap="cool",  # sequential, high contrast, visually distinct from 'viridis'
        width=0.005,  # make arrows wider (default is ~0.002)
        scale=1 / 0.1,  # adjust as needed for arrow length scaling
    )
    divider = make_axes_locatable(h_ax)
    cax1 = divider.new_horizontal(size="5%", pad=0.05, axes_class=plt.Axes)
    cax2 = divider.new_horizontal(size="5%", pad=0.75, axes_class=plt.Axes)
    h_fig.colorbar(contour, cax=cax1, label="Water Level (m)")
    h_fig.colorbar(velocity_field, cax=cax2, label="Velocity Magnitude (m/s)")
    h_fig.add_axes(cax1)
    h_fig.add_axes(cax2)
    h_ax.set_xlabel("Longitude")
    h_ax.set_ylabel("Latitude")
    h_ax.set_title(f"Wave height at frame {frame}")
    h_ax.set_xticks(np.linspace(X.min(), X.max(), 6))
    h_ax.set_yticks(np.linspace(Y.min(), Y.max(), 6))
    # if kargs.get("dark", True):
    h_fig.tight_layout()
    h_fig.savefig(
        os.path.join(output_path, f"solution_frame_{frame}.png"),
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )

    plt.show()


def animate_solution(output_path: str, frames: List[int] | None, **kargs) -> None:
    """Create an animation of solutions over multiple frames from Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    frames : List[int]
        List of frame numbers to animate.
    **kargs : dict
        Additional keyword arguments for plotting.
    """
    import matplotlib.animation as animation

    result = read_solutions(output_path, frames_list=frames)
    bathymetry = result["bathymetry"]
    X, Y = result["meshgrid"]
    solutions = result["solutions"]

    fig, ax = initialize_plot(output_path)

    wave_treshold = kargs.get("wave_treshold", 1e-2)
    max_arrow_length = kargs.get("max_arrow_length", 0.5)

    contourf = [None]
    quiver = [None]
    # colorbars = [None, None]
    divider = make_axes_locatable(ax)
    cax1 = divider.new_horizontal(size="5%", pad=0.05, axes_class=plt.Axes)
    cax2 = divider.new_horizontal(size="5%", pad=0.75, axes_class=plt.Axes)
    fig.colorbar(contourf[0], cax=cax1, label="Water Level (m)")
    fig.colorbar(quiver[0], cax=cax2, label="Velocity Magnitude (m/s)")
    fig.add_axes(cax1)
    fig.add_axes(cax2)

    def update(frame_idx: int) -> None:
        nonlocal cax1, cax2, ax

        cax1.clear()
        cax2.clear()
        ax.clear()
        # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([X.min(), X.max(), Y.min(), Y.max()], crs=ccrs.PlateCarree())
        # google_tiles = cimgt.GoogleTiles(style="street")
        # ax.add_image(google_tiles, 14, interpolation="bilinear")

        sol = solutions[frame_idx]
        h = sol[0, :, :]
        free_surface = bathymetry + h
        mask = h < wave_treshold
        free_surface[mask] = np.nan
        h_div = np.where(h <= wave_treshold, 1.0, h)
        v_x, v_y = sol[1, :, :] / h_div, sol[2, :, :] / h_div
        v_x[h <= wave_treshold] = 0.0
        v_y[h <= wave_treshold] = 0.0
        v_x_scaled, v_y_scaled = normalize_velocities_for_plotting(
            v_x, v_y, max_arrow_length
        )
        velocity = np.sqrt(v_x**2 + v_y**2)

        contourf[0] = ax.contourf(
            X, Y, free_surface, levels=50, cmap="viridis", alpha=0.7
        )
        v_x_scaled[mask] = np.nan
        v_y_scaled[mask] = np.nan
        quiver[0] = ax.quiver(
            X,
            Y,
            v_x_scaled,
            v_y_scaled,
            velocity,
            angles="xy",
            cmap="cool",
            width=0.005,
            scale=1 / 0.1,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Wave height at frame {frame_idx}")
        ax.set_xticks(np.linspace(X.min(), X.max(), 4))
        ax.set_yticks(np.linspace(Y.min(), Y.max(), 6))
        ax.set_facecolor("black")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        # Update colorbars for the new contour and quiver plots

        # Remove previous colorbars if they exist

        # divider = make_axes_locatable(ax)
        # cax1 = divider.new_horizontal(size="5%", pad=0.05, axes_class=plt.Axes)
        # cax2 = divider.new_horizontal(size="5%", pad=0.65, axes_class=plt.Axes)
        fig.colorbar(contourf[0], cax=cax1, label="Water Level (m)")
        fig.colorbar(quiver[0], cax=cax2, label="Velocity Magnitude (m/s)")

    ani = animation.FuncAnimation(
        fig, update, frames=solutions.shape[0], interval=kargs.get("interval", 200)
    )
    if kargs.get("save", False):
        ani.save(
            os.path.join(output_path, "wave_animation.mp4"),
            writer="ffmpeg",
            dpi=200,
            fps=40,
        )
    plt.show()
    return


if __name__ == "__main__":
    # plot_solution(output_path="_output", frame=20, wave_treshold=1e-2)
    animate_solution(
        output_path="_output",
        frames=list(range(0, 1000)),
        wave_treshold=1e-2,
        interval=100,
        save=False,
    )
