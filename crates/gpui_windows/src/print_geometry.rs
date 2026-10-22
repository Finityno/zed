pub(crate) const MAX_PAGES:u32=10_000;
pub(crate) const MAX_RASTER_PIXELS:u64=16_000_000;
const MAX_RASTER_SIDE:u32=16_384;
const PRINT_DPI:f64=300.;

#[derive(Clone,Copy,Debug,PartialEq)]
pub(crate) struct PrintableArea {pub width:i32,pub height:i32,pub dpi_x:i32,pub dpi_y:i32}
#[derive(Clone,Copy,Debug,PartialEq)]
pub(crate) struct PageFit {
    pub left:i32,pub top:i32,pub width:i32,pub height:i32,
    pub raster_width:u32,pub raster_height:u32,pub effective_dpi:f64,
}

pub(crate) fn selected_pages(count:u32,ranges:Option<&[(u32,u32)]>)->Result<Vec<u32>,String> {
    if count==0||count>MAX_PAGES{return Err("Print between 1 and 10,000 PDF pages.".into());}
    let Some(ranges)=ranges else{return Ok((0..count).collect());};
    if ranges.is_empty()||ranges.len()>32{return Err("Choose between 1 and 32 page ranges.".into());}
    let mut included=vec![false;count as usize];
    for &(first,last) in ranges {
        if first==0||last<first||last>count{return Err("Choose page ranges within this document.".into());}
        included[(first-1) as usize..last as usize].fill(true);
    }
    Ok(included.into_iter().enumerate().filter_map(|(index,included)|included.then_some(index as u32)).collect())
}

pub(crate) fn fit_page(width:f64,height:f64,area:PrintableArea)->Result<PageFit,String> {
    if !width.is_finite()||!height.is_finite()||width<=0.||height<=0.||area.width<=0||area.height<=0||area.dpi_x<=0||area.dpi_y<=0{return Err("The PDF page or printer has invalid dimensions.".into());}
    let printable_width=area.width as f64/area.dpi_x as f64;let printable_height=area.height as f64/area.dpi_y as f64;
    let scale=(printable_width/width).min(printable_height/height);
    let physical_width=width*scale;let physical_height=height*scale;
    if !physical_width.is_finite()||!physical_height.is_finite(){return Err("The PDF page cannot be fitted to the printer dimensions.".into());}
    let destination_width=(physical_width*area.dpi_x as f64).round().clamp(1.,area.width as f64) as i32;
    let destination_height=(physical_height*area.dpi_y as f64).round().clamp(1.,area.height as f64) as i32;
    let dpi=PRINT_DPI.min(area.dpi_x.min(area.dpi_y) as f64);
    let raster_width=(physical_width*dpi).max(1.);let raster_height=(physical_height*dpi).max(1.);
    let reduction=(MAX_RASTER_PIXELS as f64/(raster_width*raster_height)).sqrt().min(1.).min(MAX_RASTER_SIDE as f64/raster_width).min(MAX_RASTER_SIDE as f64/raster_height);
    let raster_width=(raster_width*reduction).floor().max(1.) as u32;let raster_height=(raster_height*reduction).floor().max(1.) as u32;
    if raster_width as u64*raster_height as u64>MAX_RASTER_PIXELS{return Err("The PDF page requires too many print pixels.".into());}
    Ok(PageFit{left:(area.width-destination_width)/2,top:(area.height-destination_height)/2,width:destination_width,height:destination_height,raster_width,raster_height,effective_dpi:dpi*reduction})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pdf_print_ranges_validate_merge_and_keep_document_order()->Result<(),String> {
        assert_eq!(selected_pages(4,None)?,vec![0,1,2,3]);assert_eq!(selected_pages(8,Some(&[(5,7),(2,3),(3,5)]))?,vec![1,2,3,4,5,6]);
        for ranges in [&[][..],&[(0,2)][..],&[(3,2)][..],&[(1,9)][..]]{assert!(selected_pages(8,Some(ranges)).is_err());}
        assert!(selected_pages(0,None).is_err());assert!(selected_pages(MAX_PAGES+1,None).is_err());assert!(selected_pages(10,Some(&[(1,1);33])).is_err());Ok(())
    }
    #[test]
    fn pdf_print_fit_preserves_physical_aspect_margins_and_pixel_budget()->Result<(),String> {
        let area=PrintableArea{width:4800,height:6300,dpi_x:600,dpi_y:600};let wide=fit_page(1280.,720.,area)?;
        assert_eq!((wide.left,wide.top,wide.width,wide.height),(0,1800,4800,2700));assert_eq!((wide.raster_width,wide.raster_height),(2400,1350));
        let rectangular_pixels=fit_page(100.,100.,PrintableArea{width:4800,height:3600,dpi_x:1200,dpi_y:600})?;assert_eq!(rectangular_pixels.width,4800);assert_eq!(rectangular_pixels.height,2400);assert_eq!(rectangular_pixels.top,600);
        let landscape=fit_page(1280.,720.,PrintableArea{width:6300,height:4800,..area})?;assert!(landscape.width>landscape.height&&landscape.top>0);
        let poster=fit_page(1e9,1e9,PrintableArea{width:100_000,height:100_000,dpi_x:2400,dpi_y:2400})?;assert!(poster.raster_width as u64*poster.raster_height as u64<=MAX_RASTER_PIXELS);assert!(poster.effective_dpi<PRINT_DPI);
        for (width,height) in [(0.,100.),(100.,-1.),(f64::NAN,100.),(100.,f64::INFINITY)]{assert!(fit_page(width,height,area).is_err());}
        assert!(fit_page(100.,100.,PrintableArea{dpi_x:0,..area}).is_err());Ok(())
    }
}
