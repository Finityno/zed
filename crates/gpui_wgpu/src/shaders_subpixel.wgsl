// --- subpixel sprites --- //

struct SpriteEffect {
    kind: u32,
    pad: vec3<u32>,
    bounds: Bounds,
    highlight_color: Hsla,
    band_origin: f32,
    band_width: f32,
    band_padding: vec2<f32>,
}

struct SubpixelSprite {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    color: Hsla,
    effect: SpriteEffect,
    tile: AtlasTile,
    transformation: TransformationMatrix,
}
@group(1) @binding(0) var<storage, read> b_subpixel_sprites: array<SubpixelSprite>;

struct SubpixelSpriteOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tile_position: vec2<f32>,
    @location(1) local_position: vec2<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) sprite_id: u32,
    @location(4) clip_distances: vec4<f32>,
}

struct SubpixelSpriteFragmentOutput {
    @location(0) @blend_src(0) foreground: vec4<f32>,
    @location(0) @blend_src(1) alpha: vec4<f32>,
}

@vertex
fn vs_subpixel_sprite(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> SubpixelSpriteOutput {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = b_subpixel_sprites[instance_id];

    var out = SubpixelSpriteOutput();
    out.position = to_device_position_transformed(unit_vertex, sprite.bounds, sprite.transformation);
    out.tile_position = to_tile_position(unit_vertex, sprite.tile);
    out.local_position = vec2<f32>(
        sprite.bounds.origin.x + unit_vertex.x * sprite.bounds.size.width,
        sprite.bounds.origin.y + unit_vertex.y * sprite.bounds.size.height,
    );
    out.color = hsla_to_rgba(sprite.color);
    out.sprite_id = instance_id;
    out.clip_distances = distance_from_clip_rect_transformed(unit_vertex, sprite.bounds, sprite.content_mask, sprite.transformation);
    return out;
}

fn sprite_effect_intensity(effect: SpriteEffect, local_position: vec2<f32>) -> f32 {
    switch (effect.kind) {
        default: {
            return 0.0;
        }
        case 1u: {
            let band_width = max(effect.band_width, 1.0);
            let band_start = effect.bounds.origin.x + effect.band_origin;
            let band_end = band_start + band_width;
            let feather = min(max(band_width * 0.125, 0.75), band_width * 0.5);
            let leading = saturate((local_position.x - band_start) / feather);
            let trailing = saturate((band_end - local_position.x) / feather);
            return min(leading, trailing);
        }
    }
}

@fragment
fn fs_subpixel_sprite(input: SubpixelSpriteOutput) -> SubpixelSpriteFragmentOutput {
    let sprite = b_subpixel_sprites[input.sprite_id];
    let base_color = input.color;
    let highlight_color = hsla_to_rgba(sprite.effect.highlight_color);
    let intensity = sprite_effect_intensity(sprite.effect, input.local_position);
    let sample = textureSample(t_sprite, s_sprite, input.tile_position).rgb;
    let base_alpha_corrected = apply_contrast_and_gamma_correction3(
        sample,
        base_color.rgb,
        gamma_params.subpixel_enhanced_contrast,
        gamma_params.gamma_ratios,
    );
    let highlight_alpha_corrected = apply_contrast_and_gamma_correction3(
        sample,
        highlight_color.rgb,
        gamma_params.subpixel_enhanced_contrast,
        gamma_params.gamma_ratios,
    );
    let base_alpha = base_color.a * base_alpha_corrected;
    let overlay_alpha = highlight_color.a * intensity * highlight_alpha_corrected;
    let combined_alpha = overlay_alpha + base_alpha * (vec3<f32>(1.0) - overlay_alpha);
    var combined_rgb = base_color.rgb;
    if (any(combined_alpha > vec3<f32>(0.0))) {
        combined_rgb =
            (highlight_color.rgb * overlay_alpha +
             base_color.rgb * base_alpha * (vec3<f32>(1.0) - overlay_alpha)) /
            combined_alpha;
    }

    // Alpha clip after using the derivatives.
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return SubpixelSpriteFragmentOutput(vec4<f32>(0.0), vec4<f32>(0.0));
    }

    var out = SubpixelSpriteFragmentOutput();
    out.foreground = vec4<f32>(combined_rgb, 1.0);
    out.alpha = vec4<f32>(combined_alpha, 1.0);
    return out;
}
