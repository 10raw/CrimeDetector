import React, { useState } from "react";

import { Outlet } from "react-router-dom";

import Sidebar from "../components/sidebar/sidebar.component.jsx";
import { useSelector } from "react-redux";
import { Page, Content } from "./mainLayout.styles";

import { useNavigate, useLocation } from "react-router-dom";

export const Context = React.createContext({});

const MainLayout = () => {
  const [isFullScreen, setIsFullScreen] = useState(true);
  const { isLoggedIn } = useSelector((state) => state.auth);
  const navigate = useNavigate();
  const { pathname } = useLocation();

  React.useEffect(() => {
    if (!isLoggedIn) {
      navigate("/login");
    } else {
      if (pathname === "/") {
        navigate("/");
      } else {
        navigate(pathname);
      }
    }
  }, [isLoggedIn]); 

  // React.useEffect(() => {
  //   if (!isLoggedIn) {
  //     navigate("/landing");
  //   } else {
  //     if (pathname === "/") {
  //       navigate("/home");
  //     } else {
  //       navigate(pathname);
  //     }
  //   }
  // }, [isLoggedIn]);

  return (
    <Page>
      <Context.Provider value={{ isFullScreen }}>
        <Sidebar
          toggleScreenState={() => {
            setIsFullScreen(!isFullScreen);
          }}
        />

        <Content fullScreen={isFullScreen}>
          <Outlet context={[isFullScreen]} />
        </Content>
      </Context.Provider>
    </Page>
  );
};

export default MainLayout;
